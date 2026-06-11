#!/usr/bin/env node

const fs = require("fs");
const path = require("path");
const pLimit = require("p-limit").default;

const [,, novelId, outputDir, startArg, endArg] = process.argv;

if (!novelId || !outputDir) {
    console.log(
        "Usage: node scrape.js <novel-id> <output-dir> [start] [end]\n" +
        "Examples:\n" +
        "  node scrape.js im-a-spider-so-what ./output\n" +
        "  node scrape.js im-a-spider-so-what ./output 1 100"
    );
    process.exit(1);
}

const CONCURRENCY = 5;
const MAX_RETRIES = 5;

function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

function htmlToText(html) {
    return html
        .replace(/<script[\s\S]*?<\/script>/gi, "")
        .replace(/<style[\s\S]*?<\/style>/gi, "")
        .replace(/<br\s*\/?>/gi, "\n")
        .replace(/<\/p>/gi, "\n\n")
        .replace(/<\/h[1-6]>/gi, "\n\n")
        .replace(/<[^>]+>/g, "")
        .replace(/&nbsp;/g, " ")
        .replace(/&amp;/g, "&")
        .replace(/&lt;/g, "<")
        .replace(/&gt;/g, ">")
        .replace(/\n{3,}/g, "\n\n")
        .trim();
}

// Fetch the chapter archive HTML and return ordered list of chapter-id slugs
// e.g. ["chapter-1", "chapter-2", "chapter-2001-elf-village-battle-1", ...]
async function fetchChapterIndex() {
    const url = `https://novelbin.com/ajax/chapter-archive?novelId=${encodeURIComponent(novelId)}`;
    console.log(`Fetching chapter index from ${url} ...`);

    let lastError;
    for (let attempt = 1; attempt <= MAX_RETRIES; attempt++) {
        try {
            const response = await fetch(url, {
                headers: {
                    "accept": "*/*",
                    "referer": `https://novelbin.com/b/${novelId}`,
                    "user-agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/148.0.0.0 Safari/537.36",
                    "x-requested-with": "XMLHttpRequest"
                }
            });

            if (!response.ok) throw new Error(`HTTP ${response.status}`);

            const html = await response.text();

            // Extract all href attributes from <a href="..."> in document order
            // URLs look like: https://novelbin.com/b/<novel-id>/<chapter-slug>
            const hrefRe = /href="https:\/\/novelbin\.com\/b\/[^/]+\/([^"]+)"/g;
            const slugs = [];
            let m;
            while ((m = hrefRe.exec(html)) !== null) {
                slugs.push(m[1]); // e.g. "chapter-1", "chapter-2001-elf-village-battle-1"
            }

            if (slugs.length === 0) throw new Error("No chapter links found in archive response");

            console.log(`Found ${slugs.length} chapters in index.`);
            return slugs;

        } catch (err) {
            lastError = err;
            const delay = Math.pow(2, attempt) * 1000 + Math.floor(Math.random() * 1000);
            console.log(`Index fetch retry ${attempt}/${MAX_RETRIES} in ${delay}ms: ${err.message}`);
            await sleep(delay);
        }
    }
    throw lastError;
}

async function fetchChapterContent(chapterSlug, retryLabel) {
    // The fragment API uses the full slug as chapter_id
    const url =
        `https://novelbin.com/ajax/chapter-fragment` +
        `?novel_id=${encodeURIComponent(novelId)}` +
        `&chapter_id=${encodeURIComponent(chapterSlug)}`;

    let lastError;
    for (let attempt = 1; attempt <= MAX_RETRIES; attempt++) {
        try {
            const response = await fetch(url, {
                headers: {
                    "accept": "application/json",
                    "referer": `https://novelbin.com/b/${novelId}/${chapterSlug}`,
                    "user-agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/148.0.0.0 Safari/537.36"
                }
            });

            if (!response.ok) throw new Error(`HTTP ${response.status}`);

            const data = await response.json();
            if (!data.success) throw new Error("API returned success=false");

            return data;

        } catch (err) {
            lastError = err;
            const delay = Math.pow(2, attempt) * 1000 + Math.floor(Math.random() * 1000);
            console.log(`[${retryLabel}] Retry ${attempt}/${MAX_RETRIES} in ${delay}ms`);
            await sleep(delay);
        }
    }
    throw lastError;
}

let metadata = { novel_id: novelId, novel_name: null, chapters: {} };
let metadataPath;

function saveMetadata() {
    if (!metadataPath) return;
    fs.writeFileSync(metadataPath, JSON.stringify(metadata, null, 2), "utf8");
}

async function downloadChapter(idx, chapterSlug) {
    // idx is the 1-based continuous number (1.txt, 2.txt, ...)
    const label = `${idx} (${chapterSlug})`;
    try {
        const filePath = path.join(outputDir, `${idx}.txt`);

        if (fs.existsSync(filePath)) {
            console.log(`[${label}] Skipped`);
            return;
        }

        const data = await fetchChapterContent(chapterSlug, label);
        const chapter = data.chapter;
        const text = htmlToText(chapter.content_html);

        const output = `${text}\n`;
        fs.writeFileSync(filePath, output, "utf8");

        metadata.chapters[idx] = {
            chapter_id: chapter.chapter_id,
            chapter_name: chapter.chapter_name,
            slug: chapterSlug,
            url: chapter.url
        };

        saveMetadata();
        console.log(`[${label}] Saved`);

    } catch (err) {
        console.error(`[${label}] Failed: ${err.message}`);
    }
}

async function main() {
    // 1. Fetch ordered chapter index
    const allSlugs = await fetchChapterIndex();

    // 2. Apply optional start/end range (1-based, inclusive)
    const start = startArg ? Number(startArg) : 1;
    const end   = endArg   ? Number(endArg)   : allSlugs.length;

    if (start < 1 || end > allSlugs.length || start > end) {
        console.error(
            `Range ${start}-${end} is out of bounds (index has ${allSlugs.length} chapters)`
        );
        process.exit(1);
    }

    // Slice is 0-based; chapter number is 1-based index in the full list
    const slice = allSlugs.slice(start - 1, end); // slugs to download

    // 3. Set up output directory
    fs.mkdirSync(outputDir, { recursive: true });
    metadataPath = path.join(outputDir, "metadata.json");

    if (fs.existsSync(metadataPath)) {
        try { metadata = JSON.parse(fs.readFileSync(metadataPath, "utf8")); } catch {}
    }
    metadata.novel_id = novelId;
    saveMetadata();

    console.log(`Downloading chapters ${start}–${end} into ${outputDir} ...`);

    // 4. Download with concurrency limit
    const limit = pLimit(CONCURRENCY);

    const jobs = slice.map((slug, i) => {
        const chapterNumber = start + i; // continuous 1-based number
        return limit(() => downloadChapter(chapterNumber, slug));
    });

    await Promise.all(jobs);

    saveMetadata();
    console.log("Finished.");
}

main().catch(console.error);
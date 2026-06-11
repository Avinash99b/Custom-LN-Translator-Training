#!/usr/bin/env node
const fs = require("fs");
const path = require("path");
const pLimit = require("p-limit").default;

const [, , novelSlug, startArg, endArg] = process.argv;

if (!novelSlug || !startArg || !endArg) {
    console.log(
        "Usage: node scrape-novellive.js <novel-slug> <start> <end>\n" +
        "Example:\n" +
        "node scrape-novellive.js mushoku-tensei-novel 1 500"
    );
    process.exit(1);
}

const START = Number(startArg);
const END = Number(endArg);
const CONCURRENCY = 5;
const MAX_RETRIES = 5;
const BASE_URL = "https://novellive.app";

let novelName = null;
let outputDir = null;
let metadataPath = null;

let metadata = {
    novel_id: novelSlug,
    novel_name: null,
    chapters: {}
};

function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

function sanitize(str) {
    return str.replace(/[<>:"/\\|?*]/g, "_").trim();
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

async function fetchWithRetry(chapterNum) {
    const url = `${BASE_URL}/book/${novelSlug}/chapter-${chapterNum}`;
    let lastError;

    for (let attempt = 1; attempt <= MAX_RETRIES; attempt++) {
        try {
            const response = await fetch(url, {
                headers: {

                    'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
                    'accept-language': 'en-US,en;q=0.8',
                    'priority': 'u=0, i',
                    'referer': 'https://novellive.app/book/mushoku-tensei-novel/chapter-2',
                    'sec-ch-ua': '"Chromium";v="148", "Brave";v="148", "Not/A)Brand";v="99"',
                    'sec-ch-ua-arch': '"x86"',
                    'sec-ch-ua-bitness': '"64"',
                    'sec-ch-ua-full-version-list': '"Chromium";v="148.0.0.0", "Brave";v="148.0.0.0", "Not/A)Brand";v="99.0.0.0"',
                    'sec-ch-ua-mobile': '?0',
                    'sec-ch-ua-model': '""',
                    'sec-ch-ua-platform': '"Linux"',
                    'sec-ch-ua-platform-version': '""',
                    'sec-fetch-dest': 'document',
                    'sec-fetch-mode': 'navigate',
                    'sec-fetch-site': 'same-origin',
                    'sec-fetch-user': '?1',
                    'sec-gpc': '1',
                    'upgrade-insecure-requests': '1',
                    'user-agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/148.0.0.0 Safari/537.36',
'Cookie': '_csrf=LOlngVT8UX-z1-UdXdbQBjz5; cf_clearance=VTBiZOTOIG_HlqD38GB43p0NftV1IzyjEQFbjzS.V6w-1780891923-1.2.1.1-L3zPJ2uML6qgCTXc7rhQorDZaKF62PIGzjdcgmwApyPnnnlRcSf1g9Bd6MP9ahxN.vAjnNW2j9GXfkaX5QHuIoNZNHwiRTxN4q.KBK0m956vIF8I42bWWRazZauVAOt4a9eY14OLMusvs.DcjWMqz75PFcmMdCwyyWw42X6v6HnwnBD_DUTZ3gTlBUMXVRSS.vrBuNiJWwKLKvJWNBHYsJuTvsLPxVbn4cIqY0y.oOO0vh_Xo_qrbp9Mlw1RZImNLNgiSfT8.wMMY5OPlMhpzATWYXLxKY_XIJX.I2.7x_YBWUEpA9FC2YA0M0PYGcoDRAZybkfiRdeuSTpQKH85xW6Y16z8snGk0j7ZKg2OP_JMPNPKyzNLW4eG8jv7xHajP1bJd5UgC.uOd_TI7Zf4FJfjoGhrjolrvjnEfDs3H2k'                }
            });

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}`);
            }

            return await response.text();
        } catch (err) {
            lastError = err;
            const delay = Math.pow(2, attempt) * 1000 + Math.floor(Math.random() * 1000);
            console.log(`[${chapterNum}] Retry ${attempt}/${MAX_RETRIES} in ${delay}ms`);
            await sleep(delay);
        }
    }
    throw lastError;
}
const cheerio = require("cheerio");

function extractChapterData(html) {
    const $ = cheerio.load(html);

    const novelTitle = $("h1.tit a").text().trim();
    const chapterTitle = $("span.chapter").first().text().trim();

    const contentHtml = $(".txt").html() || "";

    return {
        novel_name: novelTitle,
        chapter_name: chapterTitle,
        content_html: contentHtml
    };
}

function saveMetadata() {
    if (!metadataPath) return;
    fs.writeFileSync(metadataPath, JSON.stringify(metadata, null, 2), "utf8");
}

async function initialize() {
    const html = await fetchWithRetry(START);
    const data = extractChapterData(html);

    novelName = sanitize(data.novel_name || novelSlug);
    outputDir = path.join(process.cwd(), novelName);

    fs.mkdirSync(outputDir, { recursive: true });

    metadataPath = path.join(outputDir, "metadata.json");
    if (fs.existsSync(metadataPath)) {
        try {
            metadata = JSON.parse(fs.readFileSync(metadataPath, "utf8"));
        } catch (e) { }
    }

    metadata.novel_id = novelSlug;
    metadata.novel_name = data.novel_name || novelSlug;
    saveMetadata();

    console.log(`Output directory: ${outputDir}`);
}

async function downloadChapter(chapterNum) {
    try {
        const filePath = path.join(outputDir, `${chapterNum.toString()}.txt`);

        if (fs.existsSync(filePath)) {
            console.log(`[${chapterNum}] Skipped (already exists)`);
            return;
        }

        const html = await fetchWithRetry(chapterNum);
        const data = extractChapterData(html);

        const text = htmlToText(data.content_html);
        
        const output = `${text}`.trim();

        fs.writeFileSync(filePath, output, "utf8");

        metadata.chapters[chapterNum] = {
            chapter_name: data.chapter_name,
            url: `${BASE_URL}/book/${novelSlug}/chapter-${chapterNum}`
        };

        saveMetadata();
        console.log(`[${chapterNum}] Saved: ${data.chapter_name}`);
    } catch (err) {
        console.error(`[${chapterNum}] Failed:`, err.message);
    }
}

async function main() {
    await initialize();

    const limit = pLimit(CONCURRENCY);
    const jobs = [];

    for (let i = START; i <= END; i++) {
        jobs.push(limit(() => downloadChapter(i)));
    }

    await Promise.all(jobs);
    saveMetadata();
    console.log("✅ Finished scraping!");
}

main().catch(console.error);
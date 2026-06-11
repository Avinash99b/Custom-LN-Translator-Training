#!/usr/bin/env node

const axios = require("axios");
const cheerio = require("cheerio");
const fs = require("fs/promises");
const path = require("path");
const readline = require("readline/promises");
const pLimit = require("p-limit").default;

const CONCURRENCY = 5;

async function askNumber(rl, prompt, min, max) {
    while (true) {
        const answer = (await rl.question(prompt)).trim();

        const value = Number(answer);

        if (
            Number.isInteger(value) &&
            value >= min &&
            value <= max
        ) {
            return value;
        }

        console.log(
            `Please enter a number between ${min} and ${max}`
        );
    }
}

async function getChapterList(novelSlug) {
    const url =
        `https://novelbin.com/ajax/chapter-archive?novelId=${novelSlug}`;

    const response = await axios.get(url, {
        headers: {
            "User-Agent":
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/148.0.0.0 Safari/537.36",
            "X-Requested-With": "XMLHttpRequest",
            "Referer": `https://novelbin.com/b/${novelSlug}`,
        },
        timeout: 30000,
    });

    const $ = cheerio.load(response.data);

    const chapters = [];
    const seen = new Set();

    $("li[data-chapter-item] a").each((_, el) => {
        const href = $(el).attr("href");

        if (!href) return;

        if (seen.has(href)) return;

        seen.add(href);

        chapters.push({
            title:
                $(el).attr("title")?.trim() ||
                $(el).text().trim(),
            url: href,
        });
    });

    if (!chapters.length) {
        throw new Error(
            "No chapters found in archive."
        );
    }

    return chapters;
}

async function fetchChapter(url) {
    const response = await axios.get(url, {
        headers: {
            "User-Agent":
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/148.0.0.0 Safari/537.36",
        },
        timeout: 30000,
    });

    const $ = cheerio.load(response.data);

    const chapterContainer = $("#chr-content");

    if (!chapterContainer.length) {
        throw new Error(
            "Could not find #chr-content"
        );
    }

    chapterContainer.find("script").remove();
    chapterContainer.find("style").remove();
    chapterContainer.find(".js-ad-slot").remove();

    const paragraphs = [];

    chapterContainer.find("p").each((_, el) => {
        const text = $(el).text().trim();

        if (text) {
            paragraphs.push(text);
        }
    });

    const content =
        paragraphs.join("\n\n").trim();

    if (!content) {
        throw new Error("Chapter content empty");
    }

    return content;
}

async function main() {
    const rl = readline.createInterface({
        input: process.stdin,
        output: process.stdout,
    });

    console.log("\n==============================");
    console.log("NovelBin Downloader");
    console.log("==============================\n");

    const novelSlug = (
        await rl.question("Novel slug: ")
    ).trim();

    const outputDir = (
        await rl.question(
            "Output folder path: "
        )
    ).trim();

    console.log(
        "\nFetching chapter archive..."
    );

    const chapters =
        await getChapterList(novelSlug);

    console.log(
        `Found ${chapters.length} chapters.\n`
    );

    console.log(
        `1. ${chapters[0].title}`
    );

    console.log(
        `${chapters.length}. ${
            chapters[chapters.length - 1].title
        }\n`
    );

    const start = await askNumber(
        rl,
        `Start chapter [1-${chapters.length}]: `,
        1,
        chapters.length
    );

    const end = await askNumber(
        rl,
        `End chapter [${start}-${chapters.length}]: `,
        start,
        chapters.length
    );

    rl.close();

    await fs.mkdir(outputDir, {
        recursive: true,
    });

    await fs.writeFile(
        path.join(outputDir, "chapters.json"),
        JSON.stringify(chapters, null, 2),
        "utf8"
    );

    console.log(
        `\nDownloading chapters ${start}-${end}`
    );

    const limit = pLimit(CONCURRENCY);

    let completed = 0;

    async function downloadChapter(index) {
        const chapter = chapters[index];
        const chapterNo = index + 1;

        const filePath = path.join(
            outputDir,
            `${chapterNo}.txt`
        );

        try {
            await fs.access(filePath);

            completed++;

            console.log(
                `[${completed}/${end - start + 1}] Skip ${chapterNo}`
            );

            return;
        } catch {}

        try {
            const content =
                await fetchChapter(
                    chapter.url
                );

            await fs.writeFile(
                filePath,
                content,
                "utf8"
            );

            completed++;

            console.log(
                `[${completed}/${end - start + 1}] ✓ ${chapterNo}`
            );
        } catch (err) {
            completed++;

            console.error(
                `[${completed}/${end - start + 1}] ✗ ${chapterNo} :: ${err.message}`
            );
        }
    }

    const jobs = [];

    for (
        let index = start - 1;
        index <= end - 1;
        index++
    ) {
        jobs.push(
            limit(() =>
                downloadChapter(index)
            )
        );
    }

    await Promise.all(jobs);

    console.log("\nFinished.");
}

main().catch((err) => {
    console.error(err);
    process.exit(1);
});
#!/usr/bin/env node

/**
 * Syosetu Light Novel Chapter Scraper
 *
 * Features:
 * - Interactive CLI prompts
 * - Parallel downloads
 * - Retry support
 * - Clean TXT output
 * - Saves as chapter-0001.txt etc
 * - Resume friendly
 *
 * Usage:
 *   node scraper.js
 *
 * Install:
 *   npm init -y
 *   npm install axios cheerio p-limit prompts
 */

const axios = require("axios");
const cheerio = require("cheerio");
const fs = require("fs");
const path = require("path");
const prompts = require("prompts");
const pLimit = require("p-limit").default;

const BASE_URL = "https://ncode.syosetu.com";

function pad(num, size = 4) {
  return String(num).padStart(size, "0");
}

function sanitizeText(text) {
  return text
    .replace(/\r/g, "")
    .replace(/\u3000/g, "  ")
    .replace(/\n{3,}/g, "\n\n")
    .trim();
}

async function fetchChapter(url, retries = 3) {
  for (let attempt = 1; attempt <= retries; attempt++) {
    try {
      const response = await axios.get(url, {
        headers: {
          "User-Agent":
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome Safari",
        },
        timeout: 20000,
      });

      return response.data;
    } catch (err) {
      console.log(
        `❌ Fetch failed (${attempt}/${retries}) -> ${url}`
      );

      if (attempt === retries) {
        throw err;
      }

      await new Promise((r) => setTimeout(r, 1500));
    }
  }
}

function extractChapter(html) {
  const $ = cheerio.load(html);

  const title =
    $(".p-novel__title").first().text().trim() ||
    $("title").text().trim();

  const subtitle =
    $(".p-novel__subtitle").first().text().trim() ||
    $("h1").first().text().trim();

  const lines = [];

  $(".js-novel-text p").each((_, el) => {
    const text = $(el).text().trim();

    if (text.length === 0) {
      lines.push("");
    } else {
      lines.push(text);
    }
  });

  const body = sanitizeText(lines.join("\n"));

  return {
    title,
    subtitle,
    body,
  };
}

async function saveChapter({
  novelCode,
  chapterIndex,
  outputDir,
  retries,
}) {
  const url = `${BASE_URL}/${novelCode}/${chapterIndex}/`;

  console.log(`📥 Fetching chapter ${chapterIndex}`);

  const html = await fetchChapter(url, retries);

  const chapter = extractChapter(html);

  const fileName = `${chapterIndex}.txt`;

  const content = `
${chapter.body}
`.trim();

  fs.writeFileSync(
    path.join(outputDir, fileName),
    content,
    "utf8"
  );

  console.log(`✅ Saved ${fileName}`);
}

async function main() {
  console.log("\n📚 Syosetu Chapter Scraper\n");

  const response = await prompts([
    {
      type: "text",
      name: "novelCode",
      message: "Novel code (example: n5864cn)",
      initial: "n5864cn",
    },
    {
      type: "number",
      name: "startIndex",
      message: "Start chapter index",
      initial: 1,
      min: 1,
    },
    {
      type: "number",
      name: "endIndex",
      message: "End chapter index",
      initial: 81,
      min: 1,
    },
    {
      type: "number",
      name: "parallel",
      message: "Parallel requests",
      initial: 5,
      min: 1,
      max: 50,
    },
    {
      type: "number",
      name: "retries",
      message: "Retries per chapter",
      initial: 3,
      min: 0,
      max: 20,
    },
    {
      type: "text",
      name: "outputDir",
      message: "Output directory",
      initial: "./chapters",
    },
  ]);

  const {
    novelCode,
    startIndex,
    endIndex,
    parallel,
    retries,
    outputDir,
  } = response;

  if (!fs.existsSync(outputDir)) {
    fs.mkdirSync(outputDir, { recursive: true });
  }

  const limit = pLimit(parallel);

  const tasks = [];

  for (let i = startIndex; i <= endIndex; i++) {
    tasks.push(
      limit(async () => {
        try {
          await saveChapter({
            novelCode,
            chapterIndex: i,
            outputDir,
            retries,
          });
        } catch (err) {
          console.log(`💀 Failed chapter ${i}`);
        }
      })
    );
  }

  await Promise.all(tasks);

  console.log("\n🎉 All done!");
}

main().catch(console.error);
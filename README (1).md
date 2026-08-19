# Corporate Knowledge Base Assistant — Free Website Deployment Guide

This turns your Colab notebook into a real, shareable website — free, no credit card,
no server management. It uses **Streamlit Community Cloud**, which is built specifically
for exactly this kind of small Python + AI app.

**What the website will do:**
- A sidebar where anyone can upload PDF policy documents
- A "Build knowledge base" button (does the chunking + embedding + FAISS indexing)
- A question box on the main page
- An "Get answer" button that shows the AI's answer plus the exact source document/page

---

## Step 1 — Create a free GitHub account (skip if you already have one)

Go to https://github.com/join and sign up. GitHub is where your website's code will live —
Streamlit deploys directly from a GitHub repository.

## Step 2 — Create a new repository

1. Click the **+** icon (top right) → **New repository**
2. Name it something like `corporate-kb-rag`
3. Set it to **Public** (required for the free tier)
4. Click **Create repository**

## Step 3 — Upload these three files to the repository

From this folder, upload:
- `app.py`
- `requirements.txt`
- `README.md` (optional, just documentation)

**Easiest way (no coding tools needed):** on your new repo's GitHub page, click
**"Add file" → "Upload files"**, drag in all three files, and click **Commit changes**.

## Step 4 — Create a free Streamlit Community Cloud account

1. Go to https://streamlit.io/cloud
2. Click **Sign up**, choose **Continue with GitHub**, and authorize it
3. This is completely free for public apps

## Step 5 — Deploy the app

1. On the Streamlit Cloud dashboard, click **"Create app"**
2. Choose **"Deploy a public app from GitHub"**
3. Select:
   - Repository: `your-username/corporate-kb-rag`
   - Branch: `main`
   - Main file path: `app.py`
4. Click **Deploy**

The site will start building — this takes 1–3 minutes the first time.

## Step 6 — Add your Gemini API key (required — the app will not work without this)

The app needs your Gemini API key, but it must **never** be typed into `app.py` itself
(that would expose it publicly on GitHub). Instead:

1. On your app's page in Streamlit Cloud, click the **⋮ (three dots)** menu → **Settings**
2. Go to the **Secrets** tab
3. Paste in exactly this, replacing the placeholder with your real key:
   ```
   GEMINI_API_KEY = "your-actual-api-key-here"
   ```
4. Click **Save** — the app will automatically restart with the key available

(If you don't have a Gemini API key yet: go to https://aistudio.google.com/apikey,
sign in, and click **Create API key** — it's free.)

## Step 7 — Open and test your website

Your app will be live at a URL like:
```
https://your-username-corporate-kb-rag.streamlit.app
```

Test it:
1. Upload the 5 policy PDFs in the sidebar
2. Click **Build / Rebuild knowledge base** and wait for it to finish
3. Type a question like *"How many casual leaves am I entitled to?"*
4. Click **Get answer**

## Step 8 — Share it

That URL works for anyone, on any device, with no login required. Share it with your
MBA students or in your CIA-3 presentation as a live, working product instead of a
notebook full of code cells.

---

## Notes and limits (good to know, not blockers)

- **Free tier sleep:** if nobody visits the app for a while, Streamlit Cloud puts it to
  sleep. The next visitor just waits ~30 seconds for it to wake up — no data is lost.
- **Session-only knowledge base:** each visitor who opens the site builds their own
  knowledge base in their own browser session (nothing is shared between users, and
  nothing is saved permanently). This matches how the Colab notebook already behaves —
  it's the same in-memory design, just reachable at a public link. If you later want a
  *permanent* knowledge base that's pre-loaded for every visitor (no upload step needed),
  that's a small follow-up change — just ask.
- **Rate limits:** the free Gemini API tier has a requests-per-minute cap. The app
  already retries automatically if that's hit (same logic as the notebook's
  `rag_query_safe`).
- **Cost:** $0. GitHub, Streamlit Community Cloud, and the Gemini free tier are all free
  for this scale of usage.

import { useState, useMemo, useCallback, useRef } from "react";
import { aprioriMine } from "./apriori";
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from "recharts";

const ACCENT = "#D4A843";
const BG = "#0E1117";
const CARD = "#161B22";
const BORDER = "#2A2F38";
const TEXT = "#C9D1D9";
const MUTED = "#6B7280";
const GREEN = "#3FB950";
const RED = "#F85149";
const BLUE = "#58A6FF";

const FEATURE_COLUMNS = ["DOWN_CATEGORY", "DISTANCE_CATEGORY", "FIELD_ZONE", "OFF FORM", "BACKFIELD"];
const TARGET_MAP = { F: "F ROUTES", B: "B ROUTES", FULL: "FULL FIELD ROUTES", RP: "PLAY_TYPE_NORM" };
const RP_MODE = "RP";

const SCOUTING_SYSTEM_PROMPT = `You are an elite football defensive coordinator and film analyst. You are writing a scouting report for your coaching staff based on data-mined route tendency data from an opponent's passing game.

You will receive structured tendency data that was mined using association rule mining (Apriori algorithm) from Hudl film. Each tendency includes:
- A SITUATION (combination of down, distance, field zone, formation, backfield alignment)
- A ROUTE DISTRIBUTION (what routes they run and at what percentages)
- PLAY COUNT (sample size)
- CERTAINTY (how predictable they are — the probability of their #1 route)
- UTILITY SCORE (combines sample size and certainty — higher = more actionable)

YOUR JOB: Write a concise, actionable scouting report that a DB coach or defensive coordinator can hand to players. Follow this structure:

1. **EXECUTIVE SUMMARY** (2-3 sentences) — The single biggest takeaway. What is this offense's identity in the passing game? What is the one thing we MUST take away?

2. **HIGH-CERTAINTY TENDENCIES** — The situations where they are most predictable (certainty >= 65%). For each, state the situation, what they love to do, and a specific coverage/technique adjustment. Be specific — name leverage, cushion, eyes, bracket calls.

3. **SITUATIONAL BREAKDOWNS** — Organize by field zone or down-and-distance. Call out:
   - Red Zone tendencies
   - 3rd down tendencies
   - Backed up / coming out tendencies
   - Any formation-specific tells

4. **KEY ALERTS** — 2-3 bullet-point alerts for the game plan card. Things like "When they line up in [formation] on [down], they run [route] 75% of the time — bracket it."


STYLE RULES:
- Write like a coach, not an analyst. Direct, confident, no hedging.
- Use real football language (cover 2, MEG, bracket, robber, pattern match, carry vertical, collision, reroute, etc.)
- Reference specific routes by name when the data includes them.
- Keep it under 300 words. Coaches don't read novels.
- No intro fluff. Start with the summary immediately.`;


function parseHudlJson(data) {
  const clips = data.clipsWithTags || [];
  return clips.map((clip) => {
    const row = { clipId: clip.clipId };
    for (const tag of clip.tags || []) {
      const key = tag.key?.includes(". ") ? tag.key.split(". ").slice(1).join(". ") : tag.key;
      row[key] = tag.values?.[0] ?? null;
    }
    return row;
  });
}

function computeFieldY(yardStr) {
  try {
    const raw = parseInt(String(yardStr).trim(), 10);
    if (isNaN(raw) || raw === 0) return null;
    return raw < 0 ? Math.abs(raw) : 100 - raw;
  } catch { return null; }
}

function assignFieldZone(y) {
  if (y == null) return null;
  if (y >= 1 && y <= 15) return "Backed Up (1-15)";
  if (y >= 16 && y <= 60) return "Open Field (16-60)";
  if (y >= 61 && y <= 80) return "The Fringe (61-80)";
  if (y >= 81 && y <= 90) return "Red Zone (81-90)";
  if (y >= 91 && y <= 99) return "Gold Zone (91-99)";
  return null;
}

function assignAllFieldZones(y) {
  if (y == null) return [];
  const zones = [];
  if (y >= 1 && y <= 15) zones.push("Backed Up (1-15)");
  if (y >= 16 && y <= 60) zones.push("Open Field (16-60)");
  if (y >= 61 && y <= 80) zones.push("The Fringe (61-80)");
  if (y >= 81 && y <= 90) zones.push("Red Zone (81-90)");
  if (y >= 91 && y <= 99) zones.push("Gold Zone (91-99)");
  if (y >= 81 && y <= 99) zones.push("20 & In (81-99)");
  return zones;
}

function prepRows(rows) {
  let df = rows.map((r) => ({ ...r }));
  df.forEach((r) => {
    const dist = parseFloat(r["DIST"]);
    if (!isNaN(dist)) {
      r.DIST_NUM = dist;
      if (dist >= 1 && dist <= 3) r.DISTANCE_CATEGORY = "Short (1-3)";
      else if (dist >= 4 && dist <= 7) r.DISTANCE_CATEGORY = "Med (4-7)";
      else if (dist >= 8) r.DISTANCE_CATEGORY = "Long (8+)";
    }
    const dn = parseInt(r["DN"], 10);
    if (!isNaN(dn)) {
      r.DN_NUM = dn;
      const map = { 0: "0 DN", 1: "1st DN", 2: "2nd DN", 3: "3rd DN", 4: "4th DN" };
      r.DOWN_CATEGORY = map[dn] || null;
    }
    if (r["YARD LN"] != null) {
      const y = computeFieldY(r["YARD LN"]);
      r._fieldY = y;
      r.FIELD_ZONE = assignFieldZone(y);
    }
    for (const col of FEATURE_COLUMNS) {
      if (r[col] != null) {
        const v = String(r[col]).trim();
        if (["nan", "None", "Unknown", ""].includes(v)) r[col] = null;
        else r[col] = v;
      }
    }
    const pt = String(r["PLAY TYPE"] || "").trim().toUpperCase();
    r.PLAY_TYPE_NORM = (pt === "PASS" || pt === "SCRAMBLE") ? "PASS" : "RUN";
  });
  df = df.filter((r) => FEATURE_COLUMNS.some((c) => r[c] != null));
  return df;
}

function mineDistributions(df, viewMode, minPlays, forceFilters, fieldRange, utilityMode, certaintyMode) {
  const targetCol = TARGET_MAP[viewMode];
  let filtered = viewMode === RP_MODE
    ? [...df]
    : df.filter((r) => r.PLAY_TYPE_NORM === "PASS");

  if (fieldRange[0] > 1 || fieldRange[1] < 99) {
    filtered = filtered.filter((r) => r._fieldY != null && r._fieldY >= fieldRange[0] && r._fieldY <= fieldRange[1]);
  }

  for (const [col, val] of Object.entries(forceFilters)) {
    if (val && val !== "ALL") filtered = filtered.filter((r) => r[col] === val);
  }

  if (filtered.length < minPlays) return [];

  const transactions = filtered.map((row) => {
    const items = [];
    for (const col of FEATURE_COLUMNS) {
      if (col === "FIELD_ZONE") {
        const zones = row._fieldY != null ? assignAllFieldZones(row._fieldY) : (row[col] ? [row[col]] : []);
        for (const z of zones) items.push(`FIELD_ZONE: ${z}`);
      } else {
        if (row[col] != null) items.push(`${col}: ${row[col]}`);
      }
    }
    return items;
  });

  const frequentItemsets = aprioriMine(transactions, minPlays, 4);

  const validItemsets = frequentItemsets.filter((is) => {
    const cols = is.items.map((item) => item.split(": ")[0]);
    return new Set(cols).size === cols.length;
  });

  const K_total = Math.max(2, new Set(
    filtered.map((r) => r[targetCol]).filter((v) => v != null && !["nan", "None", ""].includes(String(v).trim()))
  ).size);

  const results = [];
  for (const { items, support } of validItemsets) {
    const conditions = items.map((item) => {
      const idx = item.indexOf(": ");
      return { col: item.slice(0, idx), val: item.slice(idx + 2) };
    });

    let subset = filtered;
    for (const { col, val } of conditions) {
      if (col === "FIELD_ZONE") {
        subset = subset.filter((r) => {
          const zones = r._fieldY != null ? assignAllFieldZones(r._fieldY) : (r[col] ? [r[col]] : []);
          return zones.includes(val);
        });
      } else {
        subset = subset.filter((r) => r[col] === val);
      }
    }

    const withTarget = subset.filter((r) => {
      const v = r[targetCol];
      return v != null && !["nan", "None", ""].includes(String(v).trim());
    });
    if (withTarget.length === 0) continue;

    const counts = {};
    withTarget.forEach((r) => { const v = String(r[targetCol]).trim(); counts[v] = (counts[v] || 0) + 1; });
    const sorted = Object.entries(counts).sort((a, b) => b[1] - a[1]);
    const total = withTarget.length;
    const primaryProb = sorted[0][1] / total;

    let certainty;
    if (certaintyMode === "entropy") {
      let H = 0;
      for (const [, c] of sorted) { const p = c / total; if (p > 0) H -= p * Math.log2(p); }
      certainty = Math.max(0, 1 - H / Math.log2(K_total));
    } else {
      certainty = primaryProb;
    }

    const distParts = sorted.map(([route, c]) => `${route} (${Math.round((c / total) * 100)}%)`);
    const distString = distParts.length > 3
      ? distParts.slice(0, 3).join(" | ") + ` | +${distParts.length - 3} more`
      : distParts.join(" | ");

    results.push({
      situation: conditions.map((c) => c.val).join(" & "),
      distribution: distString,
      plays: total,
      certainty,
      utility: (utilityMode === "log" ? Math.log(total) : Math.sqrt(total)) * certainty,
      routeBreakdown: sorted.map(([route, c]) => ({ route, count: c, pct: c / total })),
    });
  }

  results.sort((a, b) => b.utility - a.utility);
  const seen = new Set();
  return results.filter((r) => { if (seen.has(r.situation)) return false; seen.add(r.situation); return true; });
}

function getUniqueValues(df, col) {
  const vals = new Set();
  df.forEach((r) => {
    if (col === "FIELD_ZONE") {
      const zones = r._fieldY != null ? assignAllFieldZones(r._fieldY) : (r[col] ? [r[col]] : []);
      zones.forEach((z) => vals.add(z));
    } else {
      if (r[col] != null) vals.add(r[col]);
    }
  });
  return ["ALL", ...Array.from(vals).sort()];
}

function buildReportPayload(results, viewMode, totalPlays, gameLabels) {
  const top = results.slice(0, 30);
  const lines = top.map((r) =>
    `SITUATION: ${r.situation}\n  ROUTES: ${r.distribution}\n  PLAYS: ${r.plays} | CERTAINTY: ${Math.round(r.certainty * 100)}% | UTILITY: ${r.utility.toFixed(2)}`
  );
  const gamesStr = gameLabels.length > 0 ? `\nGAMES INCLUDED: ${gameLabels.join(", ")}` : "";
  return `VIEW MODE: ${viewMode} (${TARGET_MAP[viewMode]})\nTOTAL PASS PLAYS: ${totalPlays}${gamesStr}\nTOTAL TENDENCIES MINED: ${results.length}\n\n--- TOP ${top.length} TENDENCIES BY UTILITY ---\n\n${lines.join("\n\n")}`;
}

const CHART_COLORS = ["#D4A843", "#3FB950", "#58A6FF", "#F85149", "#BC8CFF", "#FF7B72", "#79C0FF", "#D2A8FF"];
const GAME_COLORS = ["#58A6FF", "#3FB950", "#BC8CFF", "#FF7B72", "#D4A843", "#79C0FF", "#F0883E", "#D2A8FF"];

const ZONE_BANDS = [
  { min: 1, max: 15, label: "Backed Up", color: "#F85149" },
  { min: 16, max: 60, label: "Open Field", color: "#58A6FF" },
  { min: 61, max: 80, label: "Fringe", color: "#BC8CFF" },
  { min: 81, max: 90, label: "Red Zone", color: "#F0883E" },
  { min: 91, max: 99, label: "Gold Zone", color: "#D4A843" },
];

function FieldRangeSlider({ range, onChange }) {
  const [lo, hi] = range;
  const pctL = ((lo - 1) / 98) * 100;
  const pctR = ((hi - 1) / 98) * 100;
  const yToLabel = (y) => y <= 50 ? `Own ${y}` : `OPP ${100 - y}`;

  return (
    <div style={{ width: "100%" }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 6 }}>
        <span style={{ fontSize: 10, color: MUTED, textTransform: "uppercase", letterSpacing: "0.05em" }}>Field Position</span>
        <span style={{ fontSize: 10, color: TEXT }}>{yToLabel(lo)} → {yToLabel(hi)}</span>
      </div>
      <div style={{ position: "relative", height: 18, borderRadius: 4, overflow: "hidden", marginBottom: 4, background: BORDER }}>
        {ZONE_BANDS.map((z) => {
          const left = ((z.min - 1) / 98) * 100;
          const width = ((z.max - z.min + 1) / 98) * 100;
          return <div key={z.label} title={z.label} style={{ position: "absolute", left: `${left}%`, width: `${width}%`, height: "100%", background: z.color, opacity: 0.35 }} />;
        })}
        <div style={{ position: "absolute", left: `${pctL}%`, width: `${pctR - pctL}%`, height: "100%", background: "rgba(255,255,255,0.18)", borderLeft: `2px solid ${ACCENT}`, borderRight: `2px solid ${ACCENT}` }} />
        {ZONE_BANDS.map((z) => {
          const center = ((z.min - 1 + (z.max - z.min) / 2) / 98) * 100;
          const width = ((z.max - z.min + 1) / 98) * 100;
          if (width < 10) return null;
          return <div key={`lbl-${z.label}`} style={{ position: "absolute", left: `${center}%`, top: "50%", transform: "translate(-50%, -50%)", fontSize: 8, color: "#FFF", fontWeight: 600, letterSpacing: "0.03em", whiteSpace: "nowrap", textShadow: "0 1px 3px rgba(0,0,0,0.6)", pointerEvents: "none" }}>{z.label}</div>;
        })}
      </div>
      <div style={{ position: "relative", height: 20 }}>
        <input type="range" min={1} max={99} value={lo}
          onChange={(e) => { const v = Math.min(Number(e.target.value), hi - 1); onChange([v, hi]); }}
          style={{ position: "absolute", top: 0, left: 0, width: "100%", height: 20, appearance: "none", background: "transparent", pointerEvents: "none", zIndex: 2, WebkitAppearance: "none" }} />
        <input type="range" min={1} max={99} value={hi}
          onChange={(e) => { const v = Math.max(Number(e.target.value), lo + 1); onChange([lo, v]); }}
          style={{ position: "absolute", top: 0, left: 0, width: "100%", height: 20, appearance: "none", background: "transparent", pointerEvents: "none", zIndex: 3, WebkitAppearance: "none" }} />
      </div>
      <div style={{ display: "flex", gap: 4, marginTop: 4, flexWrap: "wrap" }}>
        {[
          { label: "Full", r: [1, 99] },
          { label: "20 & In", r: [81, 99] },
          { label: "Backed Up", r: [1, 15] },
          { label: "Fringe+", r: [61, 99] },
        ].map((p) => (
          <button key={p.label} onClick={() => onChange(p.r)}
            style={{
              padding: "2px 7px", fontSize: 9, fontFamily: "inherit", cursor: "pointer",
              background: lo === p.r[0] && hi === p.r[1] ? `${ACCENT}30` : "transparent",
              color: lo === p.r[0] && hi === p.r[1] ? ACCENT : MUTED,
              border: `1px solid ${lo === p.r[0] && hi === p.r[1] ? ACCENT + "50" : BORDER}`,
              borderRadius: 3, whiteSpace: "nowrap",
            }}>{p.label}</button>
        ))}
      </div>
    </div>
  );
}

let _id = 1;
const uid = () => `${Date.now()}-${_id++}`;

export default function ScoutingDashboard() {
  const [games, setGames] = useState([]);
  const [expandedGame, setExpandedGame] = useState(null);
  const [editingLabel, setEditingLabel] = useState(null);
  const [newGameName, setNewGameName] = useState("");
  const [showNewGame, setShowNewGame] = useState(false);

  const [viewMode, setViewMode] = useState("B");
  const [minPlays, setMinPlays] = useState(6);
  const [filters, setFilters] = useState({});
  const [fieldRange, setFieldRange] = useState([1, 99]);
  const [sortCol, setSortCol] = useState("utility");
  const [sortAsc, setSortAsc] = useState(false);
  const [utilityMode, setUtilityMode] = useState("sqrt");
  const [certaintyMode, setCertaintyMode] = useState("top");
  const [expandedRow, setExpandedRow] = useState(null);

  const [apiKey, setApiKey] = useState("");
  const [showKey, setShowKey] = useState(false);
  const [report, setReport] = useState("");
  const [reportLoading, setReportLoading] = useState(false);
  const [reportError, setReportError] = useState("");
  const [showReport, setShowReport] = useState(false);

  const fileRefs = useRef({});

  const createGame = (label) => {
    if (!label.trim()) return;
    const id = uid();
    setGames((prev) => [...prev, { id, label: label.trim(), enabled: true, files: [] }]);
    setExpandedGame(id);
    setNewGameName("");
    setShowNewGame(false);
    setTimeout(() => { fileRefs.current[id]?.click(); }, 100);
  };

  const addFilesToGame = useCallback((gameId, fileList) => {
    Array.from(fileList).forEach((file) => {
      if (!file.name.endsWith(".json")) return;
      const reader = new FileReader();
      reader.onload = (e) => {
        try {
          const json = JSON.parse(e.target.result);
          const prepped = prepRows(parseHudlJson(json));
          setGames((prev) => prev.map((g) => g.id !== gameId ? g : { ...g, files: [...g.files, { id: uid(), fileName: file.name, rows: prepped }] }));
        } catch (err) { alert(`Error parsing ${file.name}: ${err.message}`); }
      };
      reader.readAsText(file);
    });
  }, []);

  const removeFile = (gameId, fileId) => { setGames((prev) => prev.map((g) => g.id !== gameId ? g : { ...g, files: g.files.filter((f) => f.id !== fileId) })); };
  const toggleGame = (id) => { setGames((prev) => prev.map((g) => g.id === id ? { ...g, enabled: !g.enabled } : g)); setExpandedRow(null); };
  const removeGame = (id) => { setGames((prev) => prev.filter((g) => g.id !== id)); if (expandedGame === id) setExpandedGame(null); setExpandedRow(null); };
  const updateLabel = (id, val) => { setGames((prev) => prev.map((g) => g.id === id ? { ...g, label: val || g.label } : g)); setEditingLabel(null); };

  const combinedData = useMemo(() => games.filter((g) => g.enabled).flatMap((g) => g.files.flatMap((f) => f.rows)), [games]);
  const enabledCount = games.filter((g) => g.enabled).length;

  const results = useMemo(() => {
    if (combinedData.length === 0) return [];
    return mineDistributions(combinedData, viewMode, minPlays, filters, fieldRange, utilityMode, certaintyMode);
  }, [combinedData, viewMode, minPlays, filters, fieldRange, utilityMode, certaintyMode]);

  const sortedResults = useMemo(() => {
    const r = [...results];
    r.sort((a, b) => {
      const av = a[sortCol], bv = b[sortCol];
      if (typeof av === "number") return sortAsc ? av - bv : bv - av;
      return sortAsc ? String(av).localeCompare(String(bv)) : String(bv).localeCompare(String(av));
    });
    return r.slice(0, 25);
  }, [results, sortCol, sortAsc]);

  const chartData = useMemo(() => sortedResults.slice(0, 8).map((r) => ({
    name: r.situation.length > 28 ? r.situation.slice(0, 26) + "…" : r.situation,
    utility: Math.round(r.utility * 100) / 100,
  })), [sortedResults]);

  const filterOptions = useMemo(() => {
    if (combinedData.length === 0) return {};
    const opts = {};
    for (const col of FEATURE_COLUMNS) opts[col] = getUniqueValues(combinedData, col);
    return opts;
  }, [combinedData]);

  const handleSort = (col) => { if (sortCol === col) setSortAsc(!sortAsc); else { setSortCol(col); setSortAsc(false); } };
  const sortIcon = (col) => {
    if (sortCol !== col) return <span style={{ color: MUTED, fontSize: 10, marginLeft: 4 }}>⇅</span>;
    return <span style={{ color: ACCENT, fontSize: 10, marginLeft: 4 }}>{sortAsc ? "▲" : "▼"}</span>;
  };

  const generateReport = async () => {
    if (!apiKey.trim()) { setReportError("Enter your OpenAI API key in the header first."); setShowReport(true); return; }
    if (results.length === 0) { setReportError("No tendency data. Upload game files and adjust filters."); setShowReport(true); return; }
    setReportLoading(true); setReportError(""); setReport(""); setShowReport(true);
    const enabledLabels = games.filter((g) => g.enabled).map((g) => g.label);
    const payload = buildReportPayload(results, viewMode, combinedData.length, enabledLabels);
    try {
      const res = await fetch("https://api.openai.com/v1/chat/completions", {
        method: "POST",
        headers: { "Content-Type": "application/json", Authorization: `Bearer ${apiKey.trim()}` },
        body: JSON.stringify({
          model: "gpt-4o", max_tokens: 1500, temperature: 0.4,
          messages: [
            { role: "system", content: SCOUTING_SYSTEM_PROMPT },
            { role: "user", content: `Here is the mined tendency data for this opponent's passing game. Write the scouting report.\n\n${payload}` },
          ],
        }),
      });
      if (!res.ok) { const err = await res.json().catch(() => ({})); throw new Error(err.error?.message || `API error ${res.status}`); }
      const data = await res.json();
      setReport(data.choices?.[0]?.message?.content || "No response.");
    } catch (err) { setReportError(err.message || "Failed to generate report."); }
    finally { setReportLoading(false); }
  };

  const filterLabels = { DOWN_CATEGORY: "Down", DISTANCE_CATEGORY: "Distance", FIELD_ZONE: "Field Zone", "OFF FORM": "Formation", BACKFIELD: "Backfield" };

  const renderReport = (text) => text.split("\n").map((line, i) => {
    const t = line.trim();
    if (!t) return <div key={i} style={{ height: 10 }} />;
    if (t.startsWith("**") && t.endsWith("**"))
      return <div key={i} style={{ fontSize: 13, fontWeight: 700, color: ACCENT, fontFamily: "'Space Grotesk', sans-serif", marginTop: 16, marginBottom: 6, textTransform: "uppercase", letterSpacing: "0.04em" }}>{t.replace(/\*\*/g, "")}</div>;
    if (/^\*\*[^*]+\*\*/.test(t)) {
      const parts = t.split(/\*\*/g);
      return <div key={i} style={{ fontSize: 13, lineHeight: 1.7, marginBottom: 4 }}>{parts.map((p, j) => j % 2 === 1 ? <strong key={j} style={{ color: "#FFF" }}>{p}</strong> : <span key={j}>{p}</span>)}</div>;
    }
    if (t.startsWith("- ") || t.startsWith("• "))
      return <div key={i} style={{ fontSize: 13, lineHeight: 1.7, marginBottom: 4, paddingLeft: 16 }}><span style={{ color: ACCENT, marginRight: 8 }}>▸</span>{t.slice(2)}</div>;
    if (/^#+\s/.test(t))
      return <div key={i} style={{ fontSize: 13, fontWeight: 700, color: ACCENT, fontFamily: "'Space Grotesk', sans-serif", marginTop: 16, marginBottom: 6, textTransform: "uppercase", letterSpacing: "0.04em" }}>{t.replace(/^#+\s*/, "")}</div>;
    return <div key={i} style={{ fontSize: 13, lineHeight: 1.7, marginBottom: 4 }}>{t}</div>;
  });

  const hasData = combinedData.length > 0;

  return (
    <div style={{ background: BG, minHeight: "100vh", color: TEXT, fontFamily: "'JetBrains Mono', 'Fira Code', 'SF Mono', monospace" }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');
        * { box-sizing: border-box; margin: 0; padding: 0; }
        ::-webkit-scrollbar { width: 6px; height: 6px; }
        ::-webkit-scrollbar-track { background: ${BG}; }
        ::-webkit-scrollbar-thumb { background: ${BORDER}; border-radius: 3px; }
        @keyframes pulse { 0%, 100% { opacity: 0.4; } 50% { opacity: 1; } }
        @keyframes slideDown { from { opacity: 0; transform: translateY(-8px); } to { opacity: 1; transform: translateY(0); } }
        input[type=range] { -webkit-appearance: none; pointer-events: none; }
        input[type=range]::-webkit-slider-thumb { -webkit-appearance: none; height: 16px; width: 10px; border-radius: 3px; background: ${ACCENT}; cursor: pointer; pointer-events: all; border: 1px solid ${BG}; }
        input[type=range]::-moz-range-thumb { height: 16px; width: 10px; border-radius: 3px; background: ${ACCENT}; cursor: pointer; pointer-events: all; border: 1px solid ${BG}; }
      `}</style>

      <div style={{ borderBottom: `1px solid ${BORDER}`, padding: "14px 24px", display: "flex", alignItems: "center", justifyContent: "space-between", flexWrap: "wrap", gap: 12 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 14 }}>
          <div style={{ width: 8, height: 28, background: ACCENT, borderRadius: 2 }} />
          <div>
            <div style={{ fontFamily: "'Space Grotesk', sans-serif", fontWeight: 700, fontSize: 18, color: "#FFF", letterSpacing: "-0.02em" }}>DB TENDENCY FINDER</div>
          </div>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <span style={{ fontSize: 10, color: MUTED, textTransform: "uppercase" }}>OpenAI</span>
          <div style={{ position: "relative" }}>
            <input type={showKey ? "text" : "password"} value={apiKey} onChange={(e) => setApiKey(e.target.value)} placeholder="sk-..."
              style={{ background: CARD, color: TEXT, border: `1px solid ${apiKey ? GREEN : BORDER}`, borderRadius: 6, padding: "5px 30px 5px 10px", fontSize: 11, fontFamily: "inherit", width: 200 }} />
            <button onClick={() => setShowKey(!showKey)}
              style={{ position: "absolute", right: 6, top: "50%", transform: "translateY(-50%)", background: "none", border: "none", color: MUTED, cursor: "pointer", fontSize: 13, padding: 0 }}>
              {showKey ? "◉" : "◎"}
            </button>
          </div>
          {apiKey && <div style={{ width: 6, height: 6, borderRadius: 3, background: GREEN }} />}
          {hasData && <span style={{ fontSize: 11, color: GREEN, background: "rgba(63,185,80,0.1)", padding: "3px 8px", borderRadius: 4 }}>{combinedData.length} plays</span>}
        </div>
      </div>

      <div style={{ display: "flex", minHeight: "calc(100vh - 60px)" }}>
        <div style={{ width: 280, minWidth: 280, borderRight: `1px solid ${BORDER}`, padding: "16px", display: "flex", flexDirection: "column", gap: 12 }}>
          <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
            <span style={{ fontSize: 11, color: MUTED, textTransform: "uppercase", letterSpacing: "0.06em", fontWeight: 600 }}>Games</span>
            <button onClick={() => setShowNewGame(true)}
              style={{ padding: "3px 10px", fontSize: 11, fontFamily: "inherit", background: `${ACCENT}18`, color: ACCENT, border: `1px solid ${ACCENT}40`, borderRadius: 5, cursor: "pointer", fontWeight: 600 }}>+ New Game</button>
          </div>
          {showNewGame && (
            <div style={{ display: "flex", gap: 6, animation: "slideDown 0.15s ease-out" }}>
              <input autoFocus value={newGameName} onChange={(e) => setNewGameName(e.target.value)}
                onKeyDown={(e) => { if (e.key === "Enter") createGame(newGameName); if (e.key === "Escape") { setShowNewGame(false); setNewGameName(""); } }}
                placeholder="e.g. vs Lincoln Wk3"
                style={{ flex: 1, background: CARD, color: TEXT, border: `1px solid ${ACCENT}`, borderRadius: 5, padding: "5px 8px", fontSize: 11, fontFamily: "inherit", outline: "none" }} />
              <button onClick={() => createGame(newGameName)}
                style={{ padding: "5px 10px", fontSize: 11, fontFamily: "inherit", background: ACCENT, color: BG, border: "none", borderRadius: 5, cursor: "pointer", fontWeight: 700 }}>Add</button>
            </div>
          )}
          <div style={{ flex: 1, overflowY: "auto", display: "flex", flexDirection: "column", gap: 6 }}>
            {games.length === 0 && !showNewGame && (
              <div style={{ textAlign: "center", padding: "40px 12px", color: MUTED }}>
                <div style={{ fontSize: 28, marginBottom: 10 }}>🏈</div>
                <div style={{ fontSize: 12, marginBottom: 6 }}>No games yet</div>
                <div style={{ fontSize: 11, lineHeight: 1.6 }}>Create a game, then upload<br />Hudl JSONs to it</div>
              </div>
            )}
            {games.map((g, gIdx) => {
              const gc = GAME_COLORS[gIdx % GAME_COLORS.length];
              const isOpen = expandedGame === g.id;
              const plays = g.files.reduce((s, f) => s + f.rows.length, 0);
              return (
                <div key={g.id} style={{ background: CARD, border: `1px solid ${isOpen ? gc + "50" : BORDER}`, borderRadius: 8, overflow: "hidden" }}>
                  <div style={{ display: "flex", alignItems: "center", gap: 8, padding: "8px 10px", cursor: "pointer" }} onClick={() => setExpandedGame(isOpen ? null : g.id)}>
                    <button onClick={(e) => { e.stopPropagation(); toggleGame(g.id); }}
                      style={{ width: 16, height: 16, borderRadius: 3, border: `2px solid ${gc}`, background: g.enabled ? gc : "transparent", cursor: "pointer", display: "flex", alignItems: "center", justifyContent: "center", fontSize: 9, color: "#FFF", padding: 0, flexShrink: 0 }}>
                      {g.enabled ? "✓" : ""}
                    </button>
                    <div style={{ flex: 1, minWidth: 0 }}>
                      {editingLabel === g.id ? (
                        <input autoFocus defaultValue={g.label} onClick={(e) => e.stopPropagation()}
                          onBlur={(e) => updateLabel(g.id, e.target.value)}
                          onKeyDown={(e) => { if (e.key === "Enter") updateLabel(g.id, e.target.value); if (e.key === "Escape") setEditingLabel(null); }}
                          style={{ background: "transparent", color: "#FFF", border: `1px solid ${ACCENT}`, borderRadius: 3, padding: "1px 6px", fontSize: 12, fontFamily: "inherit", width: "100%", outline: "none" }} />
                      ) : (
                        <span onDoubleClick={(e) => { e.stopPropagation(); setEditingLabel(g.id); }}
                          style={{ fontSize: 12, color: "#FFF", fontWeight: 600, opacity: g.enabled ? 1 : 0.5 }} title="Double-click to rename">{g.label}</span>
                      )}
                    </div>
                    <span style={{ fontSize: 10, color: MUTED, flexShrink: 0 }}>{g.files.length}f · {plays}p</span>
                    <span style={{ fontSize: 10, color: MUTED, transform: isOpen ? "rotate(180deg)" : "rotate(0)", transition: "transform 0.15s" }}>▾</span>
                  </div>
                  {isOpen && (
                    <div style={{ borderTop: `1px solid ${BORDER}`, padding: "8px 10px", display: "flex", flexDirection: "column", gap: 6 }}>
                      {g.files.map((f) => (
                        <div key={f.id} style={{ display: "flex", alignItems: "center", gap: 6, padding: "4px 6px", background: `${gc}08`, borderRadius: 4, fontSize: 11 }}>
                          <span style={{ color: gc, fontSize: 10 }}>◆</span>
                          <span style={{ flex: 1, color: TEXT, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{f.fileName}</span>
                          <span style={{ color: MUTED, fontSize: 10, flexShrink: 0 }}>{f.rows.length}p</span>
                          <button onClick={() => removeFile(g.id, f.id)} style={{ background: "none", border: "none", color: MUTED, cursor: "pointer", fontSize: 11, padding: 0 }}>×</button>
                        </div>
                      ))}
                      {g.files.length === 0 && <div style={{ fontSize: 11, color: MUTED, textAlign: "center", padding: 8 }}>No files yet</div>}
                      <button onClick={() => fileRefs.current[g.id]?.click()}
                        style={{ padding: "5px", fontSize: 11, fontFamily: "inherit", background: "transparent", color: gc, border: `1px dashed ${gc}40`, borderRadius: 5, cursor: "pointer", textAlign: "center" }}>+ Upload JSONs</button>
                      <input ref={(el) => (fileRefs.current[g.id] = el)} type="file" accept=".json" multiple style={{ display: "none" }}
                        onChange={(e) => { addFilesToGame(g.id, e.target.files); e.target.value = ""; }} />
                      <button onClick={() => removeGame(g.id)}
                        style={{ padding: "4px", fontSize: 10, fontFamily: "inherit", background: "transparent", color: RED + "80", border: "none", cursor: "pointer", textAlign: "center", marginTop: 2 }}>Delete Game</button>
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </div>

        <div style={{ flex: 1, padding: "16px 24px 32px", overflowY: "auto" }}>
          {!hasData && (
            <div style={{ display: "flex", justifyContent: "center", alignItems: "center", height: "60vh" }}>
              <div style={{ textAlign: "center", color: MUTED }}>
                <div style={{ fontSize: 40, marginBottom: 16 }}>📋</div>
                <div style={{ fontFamily: "'Space Grotesk', sans-serif", fontWeight: 600, fontSize: 16, color: "#FFF", marginBottom: 8 }}>Create a game and upload JSONs</div>
                <div style={{ fontSize: 12, lineHeight: 1.8 }}>Use the sidebar to add games and load Hudl exports<br />Toggle games on/off to combine data across matchups</div>
              </div>
            </div>
          )}

          {hasData && (
            <>
              <div style={{ display: "flex", gap: 10, flexWrap: "wrap", alignItems: "flex-end", marginBottom: 12 }}>
                <div>
                  <label style={{ fontSize: 10, color: MUTED, textTransform: "uppercase", letterSpacing: "0.05em", display: "block", marginBottom: 4 }}>View</label>
                  <div style={{ display: "flex" }}>
                    {[["F", "F"], ["B", "B"], ["FULL", "FULL"], ["RP", "R/P"]].map(([m, label], idx, arr) => (
                      <button key={m} onClick={() => { setViewMode(m); setExpandedRow(null); }}
                        style={{ padding: "6px 14px", fontSize: 12, fontFamily: "inherit", cursor: "pointer", background: viewMode === m ? ACCENT : CARD, color: viewMode === m ? BG : TEXT, border: `1px solid ${viewMode === m ? ACCENT : BORDER}`, borderRadius: idx === 0 ? "6px 0 0 6px" : idx === arr.length - 1 ? "0 6px 6px 0" : 0, fontWeight: viewMode === m ? 600 : 400 }}>{label}</button>
                    ))}
                  </div>
                </div>
                <div>
                  <label style={{ fontSize: 10, color: MUTED, textTransform: "uppercase", letterSpacing: "0.05em", display: "block", marginBottom: 4 }}>Min Plays: {minPlays}</label>
                  <input type="range" min={2} max={20} value={minPlays} onChange={(e) => { setMinPlays(Number(e.target.value)); setExpandedRow(null); }}
                    style={{ width: 120, accentColor: ACCENT, pointerEvents: "all" }} />
                </div>
                <div style={{ background: CARD, border: `1px solid ${BORDER}`, borderRadius: 8, padding: "8px 12px", display: "flex", flexDirection: "column", gap: 8 }}>
                  <span style={{ fontSize: 10, color: ACCENT, textTransform: "uppercase", letterSpacing: "0.06em", fontWeight: 600 }}>Utility Configuration</span>
                  <div style={{ display: "flex", gap: 12 }}>
                    <div>
                      <label style={{ fontSize: 9, color: MUTED, textTransform: "uppercase", letterSpacing: "0.05em", display: "block", marginBottom: 4 }}>n</label>
                      <div style={{ display: "flex" }}>
                        {[["sqrt", "√n"], ["log", "㏒n"]].map(([val, lbl]) => (
                          <button key={val} onClick={() => setUtilityMode(val)}
                            style={{ padding: "5px 11px", fontSize: 12, fontFamily: "inherit", cursor: "pointer", background: utilityMode === val ? ACCENT : BG, color: utilityMode === val ? BG : TEXT, border: `1px solid ${utilityMode === val ? ACCENT : BORDER}`, borderRadius: val === "sqrt" ? "6px 0 0 6px" : "0 6px 6px 0", fontWeight: utilityMode === val ? 600 : 400 }}>{lbl}</button>
                        ))}
                      </div>
                    </div>
                    <div>
                      <label style={{ fontSize: 9, color: MUTED, textTransform: "uppercase", letterSpacing: "0.05em", display: "block", marginBottom: 4 }}>Certainty</label>
                      <div style={{ display: "flex" }}>
                        {[["top", "Top"], ["entropy", "Entropy"]].map(([val, lbl]) => (
                          <button key={val} onClick={() => setCertaintyMode(val)}
                            style={{ padding: "5px 11px", fontSize: 12, fontFamily: "inherit", cursor: "pointer", background: certaintyMode === val ? ACCENT : BG, color: certaintyMode === val ? BG : TEXT, border: `1px solid ${certaintyMode === val ? ACCENT : BORDER}`, borderRadius: val === "top" ? "6px 0 0 6px" : "0 6px 6px 0", fontWeight: certaintyMode === val ? 600 : 400 }}>{lbl}</button>
                        ))}
                      </div>
                    </div>
                  </div>
                </div>
                {FEATURE_COLUMNS.filter((c) => c !== "FIELD_ZONE").map((col) => {
                  const opts = filterOptions[col];
                  if (!opts || opts.length <= 2) return null;
                  return (
                    <div key={col}>
                      <label style={{ fontSize: 10, color: MUTED, textTransform: "uppercase", letterSpacing: "0.05em", display: "block", marginBottom: 4 }}>{filterLabels[col] || col}</label>
                      <select value={filters[col] || "ALL"} onChange={(e) => { setFilters({ ...filters, [col]: e.target.value }); setExpandedRow(null); }}
                        style={{ background: CARD, color: TEXT, border: `1px solid ${BORDER}`, borderRadius: 6, padding: "6px 10px", fontSize: 12, fontFamily: "inherit", cursor: "pointer", minWidth: 120 }}>
                        {opts.map((v) => <option key={v} value={v}>{v}</option>)}
                      </select>
                    </div>
                  );
                })}
                <button onClick={() => { setFilters({}); setMinPlays(6); setFieldRange([1, 99]); setExpandedRow(null); }}
                  style={{ padding: "6px 14px", fontSize: 11, fontFamily: "inherit", background: "transparent", color: MUTED, border: `1px solid ${BORDER}`, borderRadius: 6, cursor: "pointer", alignSelf: "flex-end" }}>Reset</button>
                <button onClick={generateReport} disabled={reportLoading}
                  style={{ padding: "6px 18px", fontSize: 12, fontFamily: "inherit", cursor: reportLoading ? "wait" : "pointer", background: reportLoading ? BORDER : `linear-gradient(135deg, ${ACCENT}, #B8912E)`, color: reportLoading ? MUTED : BG, border: "none", borderRadius: 6, fontWeight: 700, alignSelf: "flex-end", display: "flex", alignItems: "center", gap: 6, opacity: reportLoading ? 0.7 : 1 }}>
                  {reportLoading ? <><span style={{ animation: "pulse 1.2s infinite" }}>◈</span> Generating...</> : <><span style={{ fontSize: 14 }}>⚡</span> AI Report</>}
                </button>
              </div>

              <div style={{ background: CARD, border: `1px solid ${BORDER}`, borderRadius: 8, padding: "12px 16px", marginBottom: 20 }}>
                <FieldRangeSlider range={fieldRange} onChange={(r) => { setFieldRange(r); setExpandedRow(null); }} />
              </div>

              {showReport && (
                <div style={{ background: CARD, border: `1px solid ${reportError ? RED : ACCENT}40`, borderRadius: 8, padding: 24, marginBottom: 20, borderLeft: `3px solid ${reportError ? RED : ACCENT}`, animation: "slideDown 0.25s ease-out" }}>
                  <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 16 }}>
                    <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                      <span style={{ fontSize: 16 }}>⚡</span>
                      <span style={{ fontSize: 12, fontWeight: 700, color: ACCENT, fontFamily: "'Space Grotesk', sans-serif", textTransform: "uppercase", letterSpacing: "0.06em" }}>AI Scouting Report — {viewMode === "RP" ? "R/P" : viewMode} View</span>
                      {report && <span style={{ fontSize: 10, color: MUTED, marginLeft: 8 }}>{enabledCount} game{enabledCount !== 1 ? "s" : ""} · GPT-4o</span>}
                    </div>
                    <div style={{ display: "flex", gap: 8 }}>
                      {report && <button onClick={() => navigator.clipboard.writeText(report)} style={{ padding: "4px 12px", fontSize: 10, fontFamily: "inherit", background: "transparent", color: MUTED, border: `1px solid ${BORDER}`, borderRadius: 4, cursor: "pointer" }}>Copy</button>}
                      <button onClick={() => setShowReport(false)} style={{ padding: "4px 10px", fontSize: 12, fontFamily: "inherit", background: "transparent", color: MUTED, border: `1px solid ${BORDER}`, borderRadius: 4, cursor: "pointer" }}>✕</button>
                    </div>
                  </div>
                  {reportError && <div style={{ color: RED, fontSize: 12, padding: "10px 14px", background: "rgba(248,81,73,0.08)", borderRadius: 6, lineHeight: 1.6 }}>{reportError}</div>}
                  {reportLoading && !report && (
                    <div style={{ color: MUTED, fontSize: 12, padding: "24px 0", textAlign: "center" }}>
                      <div style={{ animation: "pulse 1.2s infinite", fontSize: 22, marginBottom: 10 }}>◈</div>
                      Analyzing {results.length} tendencies across {combinedData.length} plays...
                    </div>
                  )}
                  {report && <div style={{ color: TEXT, maxHeight: 520, overflowY: "auto", paddingRight: 8 }}>{renderReport(report)}</div>}
                </div>
              )}

              <div style={{ display: "flex", gap: 16, marginBottom: 20 }}>
                {[
                  { label: "Tendencies", value: results.length, color: ACCENT },
                  { label: "Avg Certainty", value: results.length > 0 ? Math.round(results.reduce((s, r) => s + r.certainty, 0) / results.length * 100) + "%" : "—", color: GREEN },
                  { label: "Top Utility", value: results.length > 0 ? results[0]?.utility.toFixed(2) : "—", color: BLUE },
                ].map(({ label, value, color }) => (
                  <div key={label} style={{ background: CARD, border: `1px solid ${BORDER}`, borderRadius: 8, padding: "12px 20px", minWidth: 130 }}>
                    <div style={{ fontSize: 10, color: MUTED, textTransform: "uppercase", letterSpacing: "0.05em", marginBottom: 4 }}>{label}</div>
                    <div style={{ fontSize: 22, fontWeight: 700, color, fontFamily: "'Space Grotesk', sans-serif" }}>{value}</div>
                  </div>
                ))}
              </div>

              {chartData.length > 0 && (
                <div style={{ background: CARD, border: `1px solid ${BORDER}`, borderRadius: 8, padding: "16px 16px 8px", marginBottom: 20 }}>
                  <div style={{ fontSize: 11, color: MUTED, textTransform: "uppercase", letterSpacing: "0.06em", marginBottom: 12, paddingLeft: 4 }}>Top Situations by Utility</div>
                  <ResponsiveContainer width="100%" height={200}>
                    <BarChart data={chartData} layout="vertical" margin={{ left: 8, right: 24, top: 0, bottom: 0 }}>
                      <XAxis type="number" tick={{ fill: MUTED, fontSize: 10, fontFamily: "'JetBrains Mono', monospace" }} axisLine={{ stroke: BORDER }} tickLine={false} />
                      <YAxis type="category" dataKey="name" width={180} tick={{ fill: TEXT, fontSize: 10, fontFamily: "'JetBrains Mono', monospace" }} axisLine={false} tickLine={false} />
                      <Tooltip contentStyle={{ background: CARD, border: `1px solid ${BORDER}`, borderRadius: 8, fontFamily: "'JetBrains Mono', monospace", fontSize: 12 }} labelStyle={{ color: "#FFF", fontWeight: 600 }} formatter={(val) => [val, "Utility"]} />
                      <Bar dataKey="utility" radius={[0, 4, 4, 0]} maxBarSize={24}>
                        {chartData.map((_, i) => <Cell key={i} fill={CHART_COLORS[i % CHART_COLORS.length]} fillOpacity={0.85} />)}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              )}

              {sortedResults.length > 0 ? (
                <div style={{ background: CARD, border: `1px solid ${BORDER}`, borderRadius: 8, overflow: "hidden" }}>
                  <div style={{ overflowX: "auto" }}>
                    <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12 }}>
                      <thead>
                        <tr style={{ borderBottom: `1px solid ${BORDER}` }}>
                          {[
                            { key: "situation", label: "Situation", w: "30%" },
                            { key: "distribution", label: `${viewMode === "RP" ? "R/P" : viewMode} Distribution`, w: "34%" },
                            { key: "plays", label: "Plays", w: "9%" },
                            { key: "certainty", label: "Certainty", w: "12%" },
                            { key: "utility", label: "Utility", w: "12%" },
                          ].map(({ key, label, w }) => (
                            <th key={key} onClick={() => handleSort(key)}
                              style={{ width: w, textAlign: key === "situation" || key === "distribution" ? "left" : "center", padding: "10px 12px", fontSize: 10, color: MUTED, fontWeight: 500, textTransform: "uppercase", letterSpacing: "0.06em", cursor: "pointer", userSelect: "none", whiteSpace: "nowrap" }}>
                              {label}{sortIcon(key)}
                            </th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {sortedResults.map((r, i) => (
                          <>
                            <tr key={i} onClick={() => setExpandedRow(expandedRow === i ? null : i)}
                              style={{ borderBottom: `1px solid ${BORDER}`, background: expandedRow === i ? "rgba(212,168,67,0.05)" : i % 2 === 0 ? "transparent" : "rgba(255,255,255,0.01)", cursor: "pointer", transition: "background 0.15s" }}
                              onMouseEnter={(e) => { if (expandedRow !== i) e.currentTarget.style.background = "rgba(255,255,255,0.03)"; }}
                              onMouseLeave={(e) => { if (expandedRow !== i) e.currentTarget.style.background = i % 2 === 0 ? "transparent" : "rgba(255,255,255,0.01)"; }}>
                              <td style={{ padding: "10px 12px", fontWeight: 500, color: "#FFF" }}>{r.situation}</td>
                              <td style={{ padding: "10px 12px", color: TEXT, fontSize: 11 }}>{r.distribution}</td>
                              <td style={{ padding: "10px 12px", textAlign: "center", fontWeight: 600 }}>{r.plays}</td>
                              <td style={{ padding: "10px 12px", textAlign: "center" }}>
                                <span style={{ padding: "2px 8px", borderRadius: 4, fontSize: 11, fontWeight: 600, background: r.certainty >= 0.7 ? "rgba(63,185,80,0.12)" : r.certainty >= 0.5 ? "rgba(212,168,67,0.12)" : "rgba(248,81,73,0.1)", color: r.certainty >= 0.7 ? GREEN : r.certainty >= 0.5 ? ACCENT : RED }}>
                                  {Math.round(r.certainty * 100)}%
                                </span>
                              </td>
                              <td style={{ padding: "10px 12px", textAlign: "center", fontWeight: 600, color: ACCENT }}>{r.utility.toFixed(2)}</td>
                            </tr>
                            {expandedRow === i && (
                              <tr key={`exp-${i}`}>
                                <td colSpan={5} style={{ padding: "12px 24px 16px", background: "rgba(212,168,67,0.03)", borderBottom: `1px solid ${BORDER}` }}>
                                  <div style={{ fontSize: 10, color: MUTED, textTransform: "uppercase", marginBottom: 8, letterSpacing: "0.06em" }}>Route Breakdown</div>
                                  <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
                                    {r.routeBreakdown.map((rb, j) => (
                                      <div key={j} style={{ display: "flex", alignItems: "center", gap: 6, background: CARD, border: `1px solid ${BORDER}`, borderRadius: 6, padding: "6px 12px" }}>
                                        <div style={{ width: 60, height: 5, borderRadius: 3, background: BORDER, overflow: "hidden" }}>
                                          <div style={{ width: `${rb.pct * 100}%`, height: "100%", background: CHART_COLORS[j % CHART_COLORS.length], borderRadius: 3 }} />
                                        </div>
                                        <span style={{ fontSize: 11, color: "#FFF", fontWeight: 500 }}>{rb.route}</span>
                                        <span style={{ fontSize: 11, color: MUTED }}>{Math.round(rb.pct * 100)}%</span>
                                        <span style={{ fontSize: 10, color: MUTED }}>({rb.count})</span>
                                      </div>
                                    ))}
                                  </div>
                                </td>
                              </tr>
                            )}
                          </>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              ) : (
                <div style={{ textAlign: "center", padding: 48, color: MUTED }}>
                  <div style={{ fontSize: 32, marginBottom: 12 }}>∅</div>
                  <div style={{ fontSize: 13 }}>No tendencies found. Try lowering Min Plays or widening field range.</div>
                </div>
              )}
              <div style={{ textAlign: "center", fontSize: 10, color: MUTED, marginTop: 16 }}>Click any row to expand · Top 25 by {sortCol} </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}

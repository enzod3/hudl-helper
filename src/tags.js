// Tag config for Hudl exports. Coach tags may carry a "D. " / "DB." prefix,
// and Hudl Assist renamed several A_ fields between exports, so every value is
// read through a list of aliases.

const EMPTY = new Set(["", "nan", "None", "Unknown", "null"]);

function clean(v) {
  if (v == null) return null;
  const s = String(v).trim();
  return EMPTY.has(s) ? null : s;
}

function pick(raw, keys) {
  for (const k of keys) {
    const v = clean(raw[k]);
    if (v != null) return v;
  }
  return null;
}

function num(v) {
  const n = parseFloat(v);
  return isNaN(n) ? null : n;
}

export function normalizeKey(key) {
  if (!key) return key;
  if (key.startsWith("A_")) return key;
  return key.replace(/^[A-Z]{1,3}\.\s*/, "");
}

export function parseHudlJson(data) {
  const clips = data.clipsWithTags || [];
  return clips.map((clip) => {
    const raw = { clipId: clip.clipId };
    for (const tag of clip.tags || []) {
      const key = normalizeKey(tag.key);
      const v = tag.values?.[0] ?? null;
      // Two spellings can map to one key; never let an empty one win.
      if (clean(v) != null || raw[key] == null) raw[key] = v;
    }
    return raw;
  });
}

// ---- Field position ------------------------------------------------------

export function computeFieldY(yardStr) {
  const raw = parseInt(String(yardStr).trim(), 10);
  if (isNaN(raw) || raw === 0) return null;
  return raw < 0 ? Math.abs(raw) : 100 - raw;
}

export function assignAllFieldZones(y) {
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

// ---- Situation features (the "when") -------------------------------------
// Each derives one value (or a list, for field zone) from a parsed row.

function downLabel(r) {
  const dn = parseInt(pick(r, ["DN", "A_DOWN"]), 10);
  return { 1: "1st DN", 2: "2nd DN", 3: "3rd DN", 4: "4th DN" }[dn] || null;
}

function distLabel(r) {
  const d = num(pick(r, ["DIST", "A_YARDS_TO_GO", "A_YDSTOGO"]));
  if (d == null) return null;
  if (d >= 1 && d <= 3) return "Short (1-3)";
  if (d >= 4 && d <= 7) return "Med (4-7)";
  if (d >= 8) return "Long (8+)";
  return null;
}

function scoreLabel(r) {
  const d = num(pick(r, ["A_SCORE_DIFFERENTIAL"]));
  if (d == null) return null;
  if (d <= -9) return "Down 9+";
  if (d < 0) return "Down 1-8";
  if (d === 0) return "Tied";
  if (d <= 8) return "Up 1-8";
  return "Up 9+";
}

export const FEATURES = [
  { id: "DOWN", label: "Down", get: downLabel, on: true },
  { id: "DIST", label: "Distance", get: distLabel, on: true },
  { id: "FIELD_ZONE", label: "Field Zone", get: (r) => assignAllFieldZones(r._fieldY), on: true, multi: true },
  { id: "PERSONNEL", prefix: "Pers", label: "Personnel", get: (r) => pick(r, ["PERSONNEL"]), on: true },
  { id: "SURFACE", prefix: "Surface", label: "Surface", get: (r) => pick(r, ["SURFACE"]), on: true },
  { id: "FORM", prefix: "Form", label: "Formation", get: (r) => pick(r, ["OFF FORM"]), on: true },
  { id: "BACKFIELD", prefix: "Bkfd", label: "Backfield", get: (r) => pick(r, ["BACKFIELD"]), on: true },
  { id: "HASH", prefix: "Hash", label: "Hash", get: (r) => pick(r, ["HASH"]), on: false },
  { id: "MOTION", label: "Motion", get: (r) => (pick(r, ["MOTION"]) ? "Motion" : null), on: false },
  { id: "QTR", label: "Quarter", get: (r) => { const q = pick(r, ["A_QUARTER", "A_QTR", "QTR"]); return q ? `Q${q}` : null; }, on: false },
  { id: "SCORE", label: "Score", get: scoreLabel, on: true },
  { id: "CLOCK", label: "Clock", get: (r) => pick(r, ["A_TIME_MANAGEMENT"]), on: false },
  { id: "PREV", label: "Prev Play", get: (r) => { const p = pick(r, ["A_PLAY_AFTER"]); return p ? `After ${p}` : null; }, on: true },
  { id: "DRIVE", label: "Drive", get: (r) => (pick(r, ["A_DRIVE_FIRST_PLAY"]) === "1" ? "1st play of drive" : null), on: false },
];

// ---- Targets (the "what") -------------------------------------------------
// plays: which plays the view applies to.

export const TARGETS = [
  { id: "RP", label: "Run / Pass", plays: "ALL", get: (r) => r.PLAY_TYPE_NORM },
  { id: "CONCEPT", label: "Pass Concept", plays: "PASS", get: (r) => pick(r, ["PASS GROUP"]) },
  { id: "PASS_FAMILY", label: "Pass Family", plays: "PASS", get: (r) => pick(r, ["PASS FAMILY"]) },
  { id: "PASS_DIR", label: "Pass Direction", plays: "PASS", get: (r) => {
      const len = pick(r, ["A_PASS_LENGTH"]), loc = pick(r, ["A_PASS_LOCATION"]);
      return len && loc ? `${len} ${loc}` : null;
    } },
  { id: "TARGET_ZONE", label: "Target Zone", plays: "PASS", get: (r) => pick(r, ["TARGET ZONE"]) },
  { id: "TARGET_POS", label: "Target Position", plays: "PASS", get: (r) => pick(r, ["A_TARGETED_PLAYER_POSITION"]) },
  { id: "RB_ACTION", label: "RB Action", plays: "PASS", get: (r) => pick(r, ["RB ACTION"]) },
  { id: "F", label: "F Routes", plays: "PASS", get: (r) => pick(r, ["F ROUTES"]) },
  { id: "B", label: "B Routes", plays: "PASS", get: (r) => pick(r, ["B ROUTES"]) },
  { id: "FULL", label: "Full Field Routes", plays: "PASS", get: (r) => pick(r, ["FULL FIELD ROUTES"]) },
  { id: "RUN_FAMILY", label: "Run Family", plays: "RUN", get: (r) => pick(r, ["RUN FAMILY"]) },
  { id: "RUN", label: "Run Scheme", plays: "RUN", get: (r) => pick(r, ["RUN"]) },
  { id: "RUN_DIR", label: "Run Direction", plays: "RUN", get: (r) => pick(r, ["A_RUSH_LOCATION", "A_RUN_LOCATION"]) },
];

export const TARGET_BY_ID = Object.fromEntries(TARGETS.map((t) => [t.id, t]));

// ---- Row prep ---------------------------------------------------------------

// Success when Assist didn't supply it: 50% of the line to gain on 1st down,
// 70% on 2nd, all of it on 3rd/4th.
function computeSuccess(gain, dn, dist) {
  if (gain == null || dn == null || dist == null) return null;
  const need = dn === 1 ? 0.5 : dn === 2 ? 0.7 : 1;
  return gain >= need * dist ? 1 : 0;
}

export function prepRows(rawRows) {
  const rows = [];
  for (const raw of rawRows) {
    const pt = String(pick(raw, ["PLAY TYPE"]) || "").toUpperCase();
    // Penalties, timeouts and untagged clips aren't plays to mine.
    const PLAY_TYPE_NORM = pt === "PASS" || pt === "SCRAMBLE" ? "PASS" : pt === "RUN" ? "RUN" : null;
    if (!PLAY_TYPE_NORM) continue;

    const r = { ...raw, PLAY_TYPE_NORM };
    r._fieldY = computeFieldY(pick(raw, ["YARD LN", "A_YARDLINE", "A_YRDLN"]));
    r.OFF_TEAM = pick(raw, ["A_OFF_TEAM", "A_POSTEAM"]);
    r.ODK = pick(raw, ["ODK"]);

    for (const f of FEATURES) r[`F_${f.id}`] = f.get(r);
    for (const t of TARGETS) r[`T_${t.id}`] = t.get(r);

    const isPenalty = /penalty/i.test(pick(raw, ["RESULT"]) || "");
    const gain = num(pick(raw, ["A_YARDS_GAINED", "GN/LS"]));
    const dn = parseInt(pick(raw, ["DN", "A_DOWN"]), 10);
    const dist = num(pick(raw, ["DIST", "A_YARDS_TO_GO", "A_YDSTOGO"]));
    const assistSuccess = pick(raw, ["A_PLAY_SUCCESS", "A_SUCCESS"]);
    const firstDown = pick(raw, ["A_FIRST_DOWN"]);
    r.O_GAIN = isPenalty ? null : gain;
    r.O_SUCCESS = isPenalty ? null : assistSuccess != null ? Number(assistSuccess) : computeSuccess(gain, isNaN(dn) ? null : dn, dist);
    r.O_FIRST = isPenalty || firstDown == null ? null : Number(firstDown);
    rows.push(r);
  }
  return rows;
}

export function featureValues(r, f) {
  const v = r[`F_${f.id}`];
  if (v == null) return [];
  return Array.isArray(v) ? v : [v];
}

export function outcomeSummary(rows) {
  const avg = (key) => {
    const vals = rows.map((r) => r[key]).filter((v) => v != null);
    return vals.length ? { value: vals.reduce((s, v) => s + v, 0) / vals.length, n: vals.length } : null;
  };
  return { gain: avg("O_GAIN"), success: avg("O_SUCCESS"), first: avg("O_FIRST") };
}

export function aprioriMine(transactions, minCount, maxLen = 4) {
  const txSets = transactions.map((t) => new Set(t));

  function countSupport(itemset) {
    let count = 0;
    for (const ts of txSets) {
      if (itemset.every((item) => ts.has(item))) count++;
    }
    return count;
  }

  const results = [];

  const freq1 = new Map();
  for (const ts of txSets) {
    for (const item of ts) freq1.set(item, (freq1.get(item) || 0) + 1);
  }

  let freqSets = [];
  for (const [item, count] of freq1) {
    if (count >= minCount) {
      freqSets.push([item]);
      results.push({ items: [item], support: count });
    }
  }

  freqSets = freqSets.map((s) => [...s].sort());
  freqSets.sort((a, b) => a[0].localeCompare(b[0]));

  for (let k = 2; k <= maxLen && freqSets.length > 0; k++) {
    const candidates = [];

    for (let i = 0; i < freqSets.length; i++) {
      for (let j = i + 1; j < freqSets.length; j++) {
        const a = freqSets[i];
        const b = freqSets[j];

        let match = true;
        for (let m = 0; m < k - 2; m++) {
          if (a[m] !== b[m]) { match = false; break; }
        }
        if (!match) continue;

        candidates.push([...a.slice(0, k - 1), b[k - 2]]);
      }
    }

    const nextFreqSets = [];
    for (const candidate of candidates) {
      const count = countSupport(candidate);
      if (count >= minCount) {
        nextFreqSets.push(candidate);
        results.push({ items: candidate, support: count });
      }
    }

    freqSets = nextFreqSets;
  }

  return results;
}

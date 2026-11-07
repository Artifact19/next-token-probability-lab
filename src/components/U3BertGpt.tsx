import React, { useState, useEffect } from 'react';
import { api, BertCompareResponse } from '../api';
import { GitCompare, AlertCircle, RefreshCw, Split, Check } from 'lucide-react';

export const U3BertGpt: React.FC = () => {
  const [maskedText, setMaskedText] = useState("The doctor told the nurse that [MASK] would arrive soon.");
  const [useRightContext, setUseRightContext] = useState(true);
  const [topN, setTopN] = useState(10);
  const [data, setData] = useState<BertCompareResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleCompare = async () => {
    const maskCount = (maskedText.match(/\[MASK\]/g) || []).length;
    if (maskCount !== 1) {
      setError("Please ensure there is exactly one [MASK] token in the input sentence.");
      return;
    }
    try {
      setLoading(true);
      setError(null);
      const res = await api.compareBertGpt({
        masked_text: maskedText,
        use_right_context: useRightContext,
        top_n: topN,
      });
      setData(res);
    } catch (err: any) {
      setError(err.message || 'Comparison failed');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    handleCompare();
  }, [maskedText, useRightContext, topN]);

  return (
    <div className="space-y-8 relative">
      {/* Ambient background light glow */}
      <div className="absolute top-10 -right-10 w-96 h-96 ambient-glow-magenta pointer-events-none -z-10 opacity-50" />

      {/* Header */}
      <div className="hardware-chassis p-4 sm:p-6 rounded-2xl sm:rounded-3xl relative overflow-hidden">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-3 sm:gap-4 border-b border-white/5 pb-4 sm:pb-5">
          <div>
            <div className="flex items-center gap-2 flex-wrap">
              <span className="px-2.5 py-0.5 rounded-full text-xs font-bold bg-[#ff8a65]/20 text-[#ff8a65] border border-[#ff8a65]/40 shadow-sm">
                UNIT 03
              </span>
              <h2 className="text-lg sm:text-xl font-extrabold text-white tracking-tight">
                MASKED (BIDIRECTIONAL) VS CAUSAL (AUTOREGRESSIVE)
              </h2>
            </div>
            <p className="text-xs text-slate-400 mt-1">
              Align BERT's bidirectional [MASK] predictions against GPT-2's causal left-context next-word predictions.
            </p>
          </div>

          <div className="flex items-center gap-2 shrink-0">
            <span className="w-2.5 h-2.5 rounded-full bg-emerald-400 shadow-[0_0_10px_#34d399] animate-pulse" />
            <span className="text-xs font-mono text-emerald-300 font-semibold tracking-wider">
              BERT + GPT-2 ALIGNED
            </span>
          </div>
        </div>

        {error && (
          <div className="mt-4 p-4 bg-red-950/60 border border-red-800/80 rounded-2xl text-red-200 text-xs flex items-center gap-3 shadow-lg">
            <AlertCircle className="w-5 h-5 text-red-400 shrink-0" />
            <span>{error}</span>
          </div>
        )}

        {/* Input Deck */}
        <div className="mt-6 space-y-4">
          <div className="hardware-recess p-3.5 sm:p-4 rounded-2xl">
            <label className="block text-[11px] font-bold uppercase tracking-wider text-slate-400 mb-2">
              Cloze Sentence (Include exactly one <span className="font-mono text-fuchsia-400">[MASK]</span>)
            </label>
            <input
              type="text"
              value={maskedText}
              onChange={(e) => setMaskedText(e.target.value)}
              className="w-full bg-[#14101c] border border-white/10 rounded-xl px-4 py-3 text-xs text-slate-200 font-mono focus:outline-none focus:border-fuchsia-500 shadow-inner"
            />
          </div>

          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 sm:gap-4 pt-1">
            <div className="flex flex-wrap items-center gap-3 sm:gap-4">
              {/* Context Direction Toggle */}
              <button
                onClick={() => setUseRightContext(!useRightContext)}
                className={`px-4 py-2 rounded-xl text-xs font-bold transition-all flex items-center justify-center gap-2.5 w-full sm:w-auto cursor-pointer ${
                  useRightContext ? "btn-pill-magenta text-white" : "btn-pill-dark text-slate-400"
                }`}
              >
                <span className={`w-2 h-2 rounded-full ${useRightContext ? "bg-white shadow-[0_0_6px_#fff]" : "bg-slate-600"}`} />
                <span>Provide Right-Hand Context to BERT</span>
              </button>

              <div className="flex items-center gap-2 text-xs text-slate-400 font-medium">
                <span>Candidates:</span>
                <div className="flex gap-1">
                  {[5, 10, 15].map((n) => (
                    <button
                      key={n}
                      onClick={() => setTopN(n)}
                      className={`px-2.5 py-1 rounded-lg text-xs font-mono font-bold transition-all cursor-pointer ${
                        topN === n ? "btn-pill-magenta text-white" : "btn-pill-dark text-slate-400"
                      }`}
                    >
                      {n}
                    </button>
                  ))}
                </div>
              </div>
            </div>

            <button
              onClick={handleCompare}
              disabled={loading}
              className="btn-pill-magenta text-white px-5 py-2.5 rounded-full text-xs font-bold tracking-wide flex items-center justify-center gap-2 cursor-pointer w-full sm:w-auto"
            >
              {loading ? <RefreshCw className="w-4 h-4 animate-spin text-white" /> : <GitCompare className="w-4 h-4 text-white" />}
              <span>ALIGN PREDICTIONS</span>
            </button>
          </div>
        </div>
      </div>

      {/* Comparison Display */}
      {data && (
        <div className="space-y-6">
          {/* Jaccard Overlap Metric Card */}
          <div className="hardware-chassis p-4 sm:p-5 rounded-2xl sm:rounded-3xl flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
            <div className="flex items-center gap-3 sm:gap-4">
              <div className="w-11 h-11 sm:w-12 sm:h-12 rounded-2xl bg-gradient-to-br from-fuchsia-500/20 to-purple-500/20 border border-fuchsia-500/30 flex items-center justify-center text-fuchsia-400 shadow-md shrink-0">
                <Split className="w-5 h-5 sm:w-6 sm:h-6" />
              </div>
              <div>
                <span className="text-[10px] sm:text-[11px] font-bold uppercase tracking-wider text-slate-400 block">
                  Top-{topN} Jaccard Set Overlap
                </span>
                <span className="text-xl sm:text-2xl font-extrabold font-mono text-white">
                  {(data.jaccard_overlap * 100).toFixed(1)}% <span className="text-xs sm:text-sm text-slate-500 font-normal">({data.jaccard_overlap})</span>
                </span>
              </div>
            </div>

            <div className="flex flex-wrap items-center gap-3 sm:gap-4 text-xs font-bold">
              <span className="flex items-center gap-1.5 text-[#ff8a65]">
                <span className="w-2.5 h-2.5 rounded-full bg-[#ff8a65] shadow-[0_0_8px_#ff8a65] inline-block" />
                GPT-2 (Causal Left Context)
              </span>
              <span className="flex items-center gap-1.5 text-fuchsia-400">
                <span className="w-2.5 h-2.5 rounded-full bg-fuchsia-500 shadow-[0_0_8px_#d946ef] inline-block" />
                BERT (Bidirectional)
              </span>
            </div>
          </div>

          {/* Grouped Comparison Bars */}
          <div className="hardware-chassis p-4 sm:p-6 rounded-2xl sm:rounded-3xl space-y-4">
            <h3 className="text-sm font-bold text-white uppercase tracking-wide">
              Whole-Word Aligned Probabilities (Top Candidate Union)
            </h3>

            <div className="space-y-3 pt-2">
              {data.aligned_rows.map((row, i) => {
                const maxVal = Math.max(...data.aligned_rows.map(r => Math.max(r.gpt2_prob, r.bert_prob)), 0.05);
                return (
                  <div key={i} className="hardware-recess p-3 rounded-2xl text-xs space-y-1.5 border border-white/5">
                    <div className="flex flex-wrap justify-between items-center gap-1.5">
                      <span className="font-mono font-bold text-white bg-[#221c2e] px-2.5 py-0.5 rounded-lg border border-white/10 shadow-sm">
                        {row.word}
                      </span>
                      <div className="flex items-center gap-2 sm:gap-4 font-mono text-[10px] sm:text-[11px]">
                        <span className="text-[#ff8a65] font-bold">GPT-2: {(row.gpt2_prob * 100).toFixed(2)}%</span>
                        <span className="text-fuchsia-400 font-bold">BERT: {(row.bert_prob * 100).toFixed(2)}%</span>
                      </div>
                    </div>

                    {/* Dual visual bars */}
                    <div className="space-y-1">
                      <div className="w-full bg-[#0a0710] h-2 rounded-full overflow-hidden">
                        <div
                          className="bg-gradient-to-r from-amber-600 to-[#ff8a65] h-full rounded-full shadow-[0_0_8px_#ff8a65]"
                          style={{ width: `${(row.gpt2_prob / maxVal) * 100}%` }}
                        />
                      </div>
                      <div className="w-full bg-[#0a0710] h-2 rounded-full overflow-hidden">
                        <div
                          className="bg-gradient-to-r from-fuchsia-600 to-pink-400 h-full rounded-full shadow-[0_0_8px_#d946ef]"
                          style={{ width: `${(row.bert_prob / maxVal) * 100}%` }}
                        />
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Model Ranked Tables */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="hardware-chassis p-4 sm:p-5 rounded-2xl sm:rounded-3xl space-y-3">
              <div className="flex justify-between items-center border-b border-white/5 pb-2">
                <span className="text-xs font-bold text-[#ff8a65] uppercase tracking-wide">
                  GPT-2 (Left Context Only)
                </span>
                <span className="text-[10px] font-mono text-slate-500">Causal Mask</span>
              </div>
              <div className="overflow-x-auto">
                <table className="w-full text-left text-xs">
                  <thead>
                    <tr className="text-slate-400 border-b border-white/5 font-bold">
                      <th className="py-2">Rank</th>
                      <th className="py-2">Word</th>
                      <th className="py-2 text-right">Probability</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-white/5 font-mono">
                    {data.gpt2_top.map((item, idx) => (
                      <tr key={idx} className="hover:bg-white/[0.02]">
                        <td className="py-2 text-slate-500">#{idx + 1}</td>
                        <td className="py-2 text-white font-bold">{item.word}</td>
                        <td className="py-2 text-right text-[#ff8a65] font-extrabold">{(item.probability * 100).toFixed(2)}%</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            <div className="hardware-chassis p-4 sm:p-5 rounded-2xl sm:rounded-3xl space-y-3">
              <div className="flex justify-between items-center border-b border-white/5 pb-2">
                <span className="text-xs font-bold text-fuchsia-400 uppercase tracking-wide">
                  BERT (Both Contexts)
                </span>
                <span className="text-[10px] font-mono text-slate-500">Bidirectional</span>
              </div>
              <div className="overflow-x-auto">
                <table className="w-full text-left text-xs">
                  <thead>
                    <tr className="text-slate-400 border-b border-white/5 font-bold">
                      <th className="py-2">Rank</th>
                      <th className="py-2">Word</th>
                      <th className="py-2 text-right">Probability</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-white/5 font-mono">
                    {data.bert_top.map((item, idx) => (
                      <tr key={idx} className="hover:bg-white/[0.02]">
                        <td className="py-2 text-slate-500">#{idx + 1}</td>
                        <td className="py-2 text-white font-bold">{item.word}</td>
                        <td className="py-2 text-right text-fuchsia-400 font-extrabold">{(item.probability * 100).toFixed(2)}%</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

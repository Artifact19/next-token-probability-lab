import React, { useState, useEffect } from 'react';
import { api, AttentionResponse, PerplexityResponse } from '../api';
import { Knob } from './Knob';
import { Eye, BarChart3, AlertCircle, RefreshCw, Cpu, Database } from 'lucide-react';

export const U2Attention: React.FC = () => {
  const [prompt, setPrompt] = useState("The animal didn't cross the street because it was too tired");
  const [layer, setLayer] = useState(0); // 0-based
  const [head, setHead] = useState<string>("mean"); // "mean" or "0".."11"
  const [attentionData, setAttentionData] = useState<AttentionResponse | null>(null);
  const [attLoading, setAttLoading] = useState(false);
  const [hoveredCell, setHoveredCell] = useState<{ q: string; k: string; val: number; r: number; c: number } | null>(null);

  const [source, setSource] = useState<"wikitext" | "custom">("wikitext");
  const [customText, setCustomText] = useState("");
  const [blockSize, setBlockSize] = useState(128);
  const [pplData, setPplData] = useState<PerplexityResponse | null>(null);
  const [pplLoading, setPplLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchAttention = async () => {
    try {
      setAttLoading(true);
      setError(null);
      const headIdx = head === "mean" ? null : parseInt(head);
      const res = await api.getAttention({
        prompt,
        layer,
        head: headIdx,
      });
      setAttentionData(res);
    } catch (err: any) {
      setError(err.message || 'Failed to extract attention weights');
    } finally {
      setAttLoading(false);
    }
  };

  useEffect(() => {
    const timer = setTimeout(() => {
      fetchAttention();
    }, 300);
    return () => clearTimeout(timer);
  }, [prompt, layer, head]);

  const handleComputePpl = async () => {
    try {
      setPplLoading(true);
      setError(null);
      const res = await api.getPerplexity({
        source,
        custom_text: customText,
        block_size: blockSize,
      });
      setPplData(res);
    } catch (err: any) {
      setError(err.message || 'Failed to compute perplexity');
    } finally {
      setPplLoading(false);
    }
  };

  return (
    <div className="space-y-8 relative">
      {/* Ambient background light glow */}
      <div className="absolute top-20 -left-10 w-96 h-96 ambient-glow-magenta pointer-events-none -z-10 opacity-50" />
      <div className="absolute top-80 -right-10 w-96 h-96 ambient-glow-peach pointer-events-none -z-10 opacity-50" />

      {/* Header */}
      <div className="hardware-chassis p-6 rounded-3xl relative overflow-hidden">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-white/5 pb-5">
          <div>
            <div className="flex items-center gap-2">
              <span className="px-2.5 py-0.5 rounded-full text-xs font-bold bg-[#a855f7]/20 text-[#c084fc] border border-[#a855f7]/40 shadow-sm">
                UNIT 02
              </span>
              <h2 className="text-xl font-extrabold text-white tracking-tight">
                MULTI-HEAD SELF-ATTENTION &amp; RECURRENT BASELINE
              </h2>
            </div>
            <p className="text-xs text-slate-400 mt-1">
              Extract raw attention matrices via forward hooks across 12 transformer layers, and compare against an identical-vocabulary LSTM baseline.
            </p>
          </div>

          <div className="flex items-center gap-2">
            <span className="w-2.5 h-2.5 rounded-full bg-violet-500 shadow-[0_0_10px_#8b5cf6] animate-pulse" />
            <span className="text-xs font-mono text-violet-300 font-semibold tracking-wider">
              EAGER ATTN HOOKS
            </span>
          </div>
        </div>

        {error && (
          <div className="mt-4 p-4 bg-red-950/60 border border-red-800/80 rounded-2xl text-red-200 text-xs flex items-center gap-3 shadow-lg">
            <AlertCircle className="w-5 h-5 text-red-400 shrink-0" />
            <span>{error}</span>
          </div>
        )}

        {/* Heatmap & Attention Inspection Deck */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 mt-6">
          
          {/* Controls Column */}
          <div className="lg:col-span-4 space-y-5">
            <div className="hardware-recess p-4 rounded-2xl">
              <label className="block text-[11px] font-bold uppercase tracking-wider text-slate-400 mb-2">
                Prompt for Attention Hook
              </label>
              <textarea
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                rows={3}
                className="w-full bg-[#14101c] border border-white/10 rounded-xl p-3 text-xs text-slate-200 font-mono focus:outline-none focus:border-violet-500 transition-all resize-none shadow-inner"
              />
            </div>

            <div className="hardware-card p-5 rounded-2xl space-y-4">
              <div className="flex items-center justify-between border-b border-white/5 pb-2">
                <span className="text-xs font-bold text-slate-300">Layer Selector</span>
                <span className="text-xs font-mono font-bold text-violet-300 bg-[#0e0b14] px-2 py-0.5 rounded border border-white/5">
                  Layer {layer + 1} / 12
                </span>
              </div>
              <input
                type="range"
                min="0"
                max="11"
                value={layer}
                onChange={(e) => setLayer(parseInt(e.target.value))}
                className="slider-neon"
              />

              <div className="pt-2">
                <label className="block text-xs font-bold text-slate-300 mb-2">Attention Head</label>
                <div className="grid grid-cols-3 gap-2">
                  <button
                    onClick={() => setHead("mean")}
                    className={`py-2 px-2 rounded-xl text-[11px] font-bold transition-all text-center ${
                      head === "mean"
                        ? "btn-pill-magenta text-white"
                        : "btn-pill-dark text-slate-400"
                    }`}
                  >
                    Mean Heads
                  </button>
                  {[0, 1, 2, 3, 4, 5, 6, 7].map((h) => (
                    <button
                      key={h}
                      onClick={() => setHead(h.toString())}
                      className={`py-2 px-2 rounded-xl text-[11px] font-mono font-bold transition-all text-center ${
                        head === h.toString()
                          ? "btn-pill-magenta text-white"
                          : "btn-pill-dark text-slate-400"
                      }`}
                    >
                      Head {h}
                    </button>
                  ))}
                </div>
              </div>
            </div>
          </div>

          {/* Heatmap Visualizer */}
          <div className="lg:col-span-8 space-y-5">
            <div className="hardware-card p-5 rounded-2xl space-y-4">
              <div className="flex items-center justify-between border-b border-white/5 pb-3">
                <span className="text-xs font-bold text-white flex items-center gap-2">
                  <Eye className="w-4 h-4 text-violet-400" />
                  Self-Attention Matrix: Query Rows &times; Key Columns
                </span>
                <span className="text-[11px] text-slate-400 font-mono">
                  Rows sum to 1.0 (Softmax)
                </span>
              </div>

              {attLoading && !attentionData ? (
                <div className="h-64 flex items-center justify-center text-slate-500">
                  <RefreshCw className="w-6 h-6 animate-spin text-violet-400" />
                </div>
              ) : attentionData ? (
                <div className="space-y-4">
                  {/* Heatmap Table */}
                  <div className="overflow-x-auto pb-4 pt-6 hardware-recess p-4 rounded-2xl scrollbar-thin">
                    <div className="sm:hidden flex items-center gap-1.5 text-[10px] text-fuchsia-400/80 font-mono mb-3">
                      <span>&larr; Swipe matrix horizontally to view all tokens &rarr;</span>
                    </div>

                    <div className="inline-block min-w-full">
                      {/* Key tokens header */}
                      <div className="flex items-end pl-24 gap-1 h-28 mb-3 overflow-visible">
                        {attentionData.tokens.map((tok, c) => (
                          <div
                            key={c}
                            className="w-11 flex-shrink-0 flex items-end justify-center h-full relative"
                          >
                            <span
                              className="block transform -rotate-45 origin-bottom-left text-[11px] font-mono text-slate-300 font-medium whitespace-nowrap select-none hover:text-fuchsia-300 transition-colors pointer-events-auto cursor-default pl-1"
                              title={tok}
                            >
                              {tok}
                            </span>
                          </div>
                        ))}
                      </div>

                      {/* Query token rows */}
                      <div className="space-y-1">
                        {attentionData.tokens.map((qTok, r) => (
                          <div key={r} className="flex items-center">
                            <div className="w-24 text-[11px] font-mono text-right pr-3 text-slate-300 truncate" title={qTok}>
                              {qTok}
                            </div>
                            <div className="flex gap-1">
                              {attentionData.tokens.map((kTok, c) => {
                                const val = attentionData.matrix[r]?.[c] ?? 0;
                                return (
                                  <div
                                    key={c}
                                    onMouseEnter={() => setHoveredCell({ q: qTok, k: kTok, val, r, c })}
                                    onMouseLeave={() => setHoveredCell(null)}
                                    className="w-11 h-8 rounded-md transition-all duration-150 cursor-pointer flex items-center justify-center relative group"
                                    style={{
                                      backgroundColor: val > 0 ? `rgba(217, 70, 239, ${Math.max(0.08, val * 0.95)})` : '#120f18',
                                      border: '1px solid rgba(255, 255, 255, 0.05)',
                                      boxShadow: val > 0.3 ? '0 0 10px rgba(217, 70, 239, 0.35)' : 'none',
                                    }}
                                  >
                                    {val > 0.15 && (
                                      <span className="text-[10px] font-mono font-bold text-white select-none">
                                        {val.toFixed(2)}
                                      </span>
                                    )}
                                  </div>
                                );
                              })}
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>

                  {/* Hover Cell Inspector */}
                  <div className="flex items-center justify-between p-3.5 hardware-recess rounded-xl text-xs">
                    <div>
                      {hoveredCell ? (
                        <span className="font-mono text-slate-200">
                          Query <span className="text-fuchsia-400 font-bold bg-[#1d1627] px-1.5 py-0.5 rounded">{hoveredCell.q}</span> &rarr; Key <span className="text-amber-400 font-bold bg-[#1d1627] px-1.5 py-0.5 rounded">{hoveredCell.k}</span>: Weight <span className="text-white font-bold">{hoveredCell.val.toFixed(4)}</span>
                        </span>
                      ) : (
                        <span className="text-slate-500 italic">Hover over any matrix cell to read exact attention weight</span>
                      )}
                    </div>
                    <span className="text-[11px] font-mono text-slate-400">
                      Layer {layer + 1} &bull; {head === 'mean' ? 'Mean 12 Heads' : `Head ${head}`}
                    </span>
                  </div>
                </div>
              ) : null}
            </div>
          </div>
        </div>
      </div>

      {/* Recurrent Baseline Perplexity Chamber */}
      <div className="hardware-chassis p-6 rounded-3xl space-y-5">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-white/5 pb-4">
          <div>
            <h3 className="text-base font-bold text-white flex items-center gap-2">
              <BarChart3 className="w-5 h-5 text-emerald-400" />
              Perplexity Benchmark: 2-Layer LSTM vs Pre-Trained GPT-2
            </h3>
            <p className="text-xs text-slate-400">
              Evaluated on identical non-overlapping blocks over the GPT-2 BPE tokenizer vocabulary.
            </p>
          </div>

          <button
            onClick={handleComputePpl}
            disabled={pplLoading}
            className="btn-pill-magenta text-white px-5 py-2.5 rounded-full text-xs font-bold tracking-wide flex items-center gap-2 cursor-pointer self-start sm:self-auto"
          >
            {pplLoading ? <RefreshCw className="w-4 h-4 animate-spin text-white" /> : <Database className="w-4 h-4 text-white" />}
            <span>COMPUTE PERPLEXITY</span>
          </button>
        </div>

        {/* Options */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="hardware-recess p-4 rounded-2xl flex items-center justify-between">
            <span className="text-xs font-bold text-slate-300">Evaluation Corpus</span>
            <div className="flex gap-2">
              <button
                onClick={() => setSource("wikitext")}
                className={`px-3 py-1.5 rounded-xl text-xs font-bold transition-all ${
                  source === "wikitext" ? "btn-pill-magenta text-white" : "btn-pill-dark text-slate-400"
                }`}
              >
                WikiText-2 Test
              </button>
              <button
                onClick={() => setSource("custom")}
                className={`px-3 py-1.5 rounded-xl text-xs font-bold transition-all ${
                  source === "custom" ? "btn-pill-magenta text-white" : "btn-pill-dark text-slate-400"
                }`}
              >
                Custom Text
              </button>
            </div>
          </div>

          <div className="hardware-recess p-4 rounded-2xl flex items-center justify-between">
            <span className="text-xs font-bold text-slate-300">Context Window Block Size</span>
            <div className="flex gap-2">
              {[64, 128, 256].map((b) => (
                <button
                  key={b}
                  onClick={() => setBlockSize(b)}
                  className={`px-3 py-1.5 rounded-xl text-xs font-mono font-bold transition-all ${
                    blockSize === b ? "btn-pill-magenta text-white" : "btn-pill-dark text-slate-400"
                  }`}
                >
                  {b}
                </button>
              ))}
            </div>
          </div>
        </div>

        {source === "custom" && (
          <div className="hardware-recess p-4 rounded-2xl">
            <textarea
              value={customText}
              onChange={(e) => setCustomText(e.target.value)}
              rows={3}
              placeholder="Enter custom text for held-out block perplexity evaluation..."
              className="w-full bg-[#14101c] border border-white/10 rounded-xl p-3 text-xs text-slate-200 font-mono focus:outline-none focus:border-violet-500 resize-none shadow-inner"
            />
          </div>
        )}

        {pplData && (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 pt-2">
            <div className="hardware-card p-5 rounded-2xl space-y-4">
              <span className="text-xs font-bold text-slate-300 block uppercase tracking-wide">
                Perplexity (Logarithmic Scale)
              </span>

              <div className="space-y-4 pt-1">
                {pplData.lstm_ppl && (
                  <div>
                    <div className="flex justify-between text-xs mb-1.5">
                      <span className="text-[#ff8a65] font-bold">LSTM Baseline (Tied Embeddings)</span>
                      <span className="font-mono text-white font-extrabold">{pplData.lstm_ppl.toFixed(1)}</span>
                    </div>
                    <div className="w-full bg-[#0d0a13] h-3 rounded-full overflow-hidden p-0.5 border border-white/5">
                      <div className="bg-gradient-to-r from-amber-600 to-[#ff8a65] h-full rounded-full shadow-[0_0_8px_#ff8a65]" style={{ width: '90%' }} />
                    </div>
                  </div>
                )}

                <div>
                  <div className="flex justify-between text-xs mb-1.5">
                    <span className="text-fuchsia-400 font-bold">GPT-2 Pre-Trained Transformer</span>
                    <span className="font-mono text-white font-extrabold">{pplData.gpt_ppl.toFixed(1)}</span>
                  </div>
                  <div className="w-full bg-[#0d0a13] h-3 rounded-full overflow-hidden p-0.5 border border-white/5">
                    <div className="bg-gradient-to-r from-fuchsia-600 to-pink-400 h-full rounded-full shadow-[0_0_8px_#d946ef]" style={{ width: '28%' }} />
                  </div>
                </div>
              </div>
            </div>

            <div className="hardware-card p-5 rounded-2xl flex flex-col justify-between">
              <div>
                <span className="text-xs font-bold text-slate-300 block uppercase tracking-wide mb-3">
                  Summary Table
                </span>
                <table className="w-full text-left text-xs border-collapse">
                  <thead>
                    <tr className="border-b border-white/10 text-slate-400 font-bold">
                      <th className="py-2 px-2">Model Architecture</th>
                      <th className="py-2 px-2">Perplexity</th>
                      <th className="py-2 px-2">Tokens</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-white/5 font-mono">
                    {pplData.lstm_ppl && (
                      <tr>
                        <td className="py-2 px-2 text-[#ff8a65] font-bold">LSTM Baseline</td>
                        <td className="py-2 px-2 text-white font-extrabold">{pplData.lstm_ppl.toFixed(1)}</td>
                        <td className="py-2 px-2 text-slate-400">{pplData.token_count}</td>
                      </tr>
                    )}
                    <tr>
                      <td className="py-2 px-2 text-fuchsia-400 font-bold">GPT-2 (Causal)</td>
                      <td className="py-2 px-2 text-white font-extrabold">{pplData.gpt_ppl.toFixed(1)}</td>
                      <td className="py-2 px-2 text-slate-400">{pplData.token_count}</td>
                    </tr>
                  </tbody>
                </table>
              </div>
              <span className="text-[11px] text-slate-500 mt-4 block">
                Non-overlapping {blockSize}-token evaluation blocks with causal one-token prediction shift.
              </span>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

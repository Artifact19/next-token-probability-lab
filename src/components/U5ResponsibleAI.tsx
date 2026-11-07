import React, { useState } from 'react';
import { api, AuditReport, ToxicSuppressionResponse, CalibrationResponse } from '../api';
import { ShieldCheck, AlertTriangle, Scale, BookOpen, RefreshCw, ChevronDown, ChevronUp } from 'lucide-react';

export const U5ResponsibleAI: React.FC = () => {
  const [samplesPerItem, setSamplesPerItem] = useState(2);
  const [auditReports, setAuditReports] = useState<AuditReport[] | null>(null);
  const [auditing, setAuditing] = useState(false);
  const [expandedFailures, setExpandedFailures] = useState<number | null>(null);

  const [toxicPrompt, setToxicPrompt] = useState("You are such a");
  const [toxicTemp, setToxicTemp] = useState(1.2);
  const [toxicSeed, setToxicSeed] = useState(3);
  const [toxicData, setToxicData] = useState<ToxicSuppressionResponse | null>(null);
  const [toxicLoading, setToxicLoading] = useState(false);

  const [calibData, setCalibData] = useState<CalibrationResponse | null>(null);
  const [calibLoading, setCalibLoading] = useState(false);

  const [modelCardContent, setModelCardContent] = useState<string | null>(null);
  const [showModelCard, setShowModelCard] = useState(false);
  const [hoveredCalibBin, setHoveredCalibBin] = useState<{ bin: number; conf: number; acc: number; count: number } | null>(null);

  const [error, setError] = useState<string | null>(null);

  const handleRunAudit = async () => {
    try {
      setAuditing(true);
      setError(null);
      const res = await api.runAudit({
        temperatures: [0.2, 1.2],
        samples_per_item: samplesPerItem,
      });
      setAuditReports(res.reports);
    } catch (err: any) {
      setError(err.message || 'Audit run failed');
    } finally {
      setAuditing(false);
    }
  };

  const handleTestToxic = async () => {
    try {
      setToxicLoading(true);
      setError(null);
      const res = await api.testToxicFilter({
        prompt: toxicPrompt,
        temperature: toxicTemp,
        seed: toxicSeed,
        max_tokens: 30,
      });
      setToxicData(res);
    } catch (err: any) {
      setError(err.message || 'Toxic filter test failed');
    } finally {
      setToxicLoading(false);
    }
  };

  const handleComputeCalibration = async () => {
    try {
      setCalibLoading(true);
      setError(null);
      const res = await api.getCalibration();
      setCalibData(res);
    } catch (err: any) {
      setError(err.message || 'Calibration computation failed');
    } finally {
      setCalibLoading(false);
    }
  };

  const handleLoadModelCard = async () => {
    if (modelCardContent) {
      setShowModelCard(!showModelCard);
      return;
    }
    try {
      const res = await api.getModelCard();
      setModelCardContent(res.content);
      setShowModelCard(true);
    } catch (err: any) {
      setError(err.message || 'Failed to load model card');
    }
  };

  return (
    <div className="space-y-8 relative">
      {/* Ambient background light glow */}
      <div className="absolute top-10 -right-10 w-96 h-96 ambient-glow-magenta pointer-events-none -z-10 opacity-50" />
      <div className="absolute top-80 -left-10 w-96 h-96 ambient-glow-peach pointer-events-none -z-10 opacity-50" />

      {/* Header */}
      <div className="hardware-chassis p-6 rounded-3xl relative overflow-hidden">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-white/5 pb-5">
          <div>
            <div className="flex items-center gap-2">
              <span className="px-2.5 py-0.5 rounded-full text-xs font-bold bg-[#ff8a65]/20 text-[#ff8a65] border border-[#ff8a65]/40 shadow-sm">
                UNIT 05
              </span>
              <h2 className="text-xl font-extrabold text-white tracking-tight">
                RESPONSIBLE AI, GUARDRAILS &amp; HALLUCINATION AUDIT
              </h2>
            </div>
            <p className="text-xs text-slate-400 mt-1">
              Audit factual error rates under pure temperature sampling, enforce real-time logit masking (-inf), and measure calibration.
            </p>
          </div>

          <div className="flex items-center gap-2">
            <span className="w-2.5 h-2.5 rounded-full bg-emerald-400 shadow-[0_0_10px_#34d399] animate-pulse" />
            <span className="text-xs font-mono text-emerald-300 font-semibold tracking-wider">
              SAFETY EVALUATION ACTIVE
            </span>
          </div>
        </div>

        {error && (
          <div className="mt-4 p-4 bg-red-950/60 border border-red-800/80 rounded-2xl text-red-200 text-xs flex items-center gap-3 shadow-lg">
            <AlertTriangle className="w-5 h-5 text-red-400 shrink-0" />
            <span>{error}</span>
          </div>
        )}

        {/* 1. Hallucination Audit Section */}
        <div className="mt-6 space-y-4">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-white/5 pb-3">
            <div>
              <h3 className="text-sm font-bold text-white flex items-center gap-2">
                <AlertTriangle className="w-4 h-4 text-amber-400" />
                Closed-Book Cloze Factual Audit (24 Questions)
              </h3>
              <p className="text-xs text-slate-400">
                Measures error rate and unsupported named-entity / numeric hallucination rates with 95% Wilson intervals.
              </p>
            </div>

            <div className="flex items-center gap-3">
              <span className="text-xs text-slate-400 font-medium">Samples / Q:</span>
              <div className="flex gap-1">
                {[1, 2, 3].map((s) => (
                  <button
                    key={s}
                    onClick={() => setSamplesPerItem(s)}
                    className={`px-2.5 py-1 rounded-lg text-xs font-mono font-bold transition-all ${
                      samplesPerItem === s ? "btn-pill-magenta text-white" : "btn-pill-dark text-slate-400"
                    }`}
                  >
                    {s}
                  </button>
                ))}
              </div>
              <button
                onClick={handleRunAudit}
                disabled={auditing}
                className="btn-pill-magenta text-white px-5 py-2.5 rounded-full text-xs font-bold tracking-wide flex items-center gap-2 cursor-pointer ml-2"
              >
                {auditing ? <RefreshCw className="w-3.5 h-3.5 animate-spin" /> : <ShieldCheck className="w-3.5 h-3.5" />}
                <span>RUN AUDIT</span>
              </button>
            </div>
          </div>

          {auditReports && (
            <div className="space-y-4 pt-2">
              <div className="hardware-recess p-2 rounded-2xl overflow-x-auto">
                <table className="w-full text-left text-xs border-collapse">
                  <thead>
                    <tr className="border-b border-white/10 text-slate-400 font-bold">
                      <th className="py-2.5 px-3">Temperature</th>
                      <th className="py-2.5 px-3">Trials</th>
                      <th className="py-2.5 px-3">Factual Error Rate (95% CI)</th>
                      <th className="py-2.5 px-3">Hallucination Rate (95% CI)</th>
                      <th className="py-2.5 px-3">Step Entropy</th>
                      <th className="py-2.5 px-3 text-right">Failures</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-white/5 font-mono text-slate-300">
                    {auditReports.map((r, idx) => (
                      <React.Fragment key={idx}>
                        <tr className="hover:bg-white/[0.02]">
                          <td className="py-2.5 px-3 font-extrabold text-white">T = {r.temperature.toFixed(1)}</td>
                          <td className="py-2.5 px-3">{r.trials}</td>
                          <td className="py-2.5 px-3 text-[#ff8a65] font-bold">
                            {(r.error_rate * 100).toFixed(1)}% ({(r.error_ci[0] * 100).toFixed(1)}%-{(r.error_ci[1] * 100).toFixed(1)}%)
                          </td>
                          <td className="py-2.5 px-3 text-fuchsia-400 font-bold">
                            {(r.hallucination_rate * 100).toFixed(1)}% ({(r.hallucination_ci[0] * 100).toFixed(1)}%-{(r.hallucination_ci[1] * 100).toFixed(1)}%)
                          </td>
                          <td className="py-2.5 px-3 text-violet-400">{r.mean_entropy_bits.toFixed(2)} bits</td>
                          <td className="py-2.5 px-3 text-right">
                            <button
                              onClick={() => setExpandedFailures(expandedFailures === idx ? null : idx)}
                              className="btn-pill-dark px-3 py-1 rounded-lg text-[11px] font-sans font-bold text-slate-300 inline-flex items-center gap-1"
                            >
                              <span>Inspect ({r.failures.length})</span>
                              {expandedFailures === idx ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
                            </button>
                          </td>
                        </tr>
                        {expandedFailures === idx && (
                          <tr>
                            <td colSpan={6} className="p-4 bg-[#14101c] border-b border-white/5">
                              <div className="space-y-2">
                                <span className="text-xs font-bold text-slate-300 block font-sans">
                                  Failure Case Inspection (T = {r.temperature.toFixed(1)})
                                </span>
                                <div className="grid grid-cols-1 md:grid-cols-2 gap-2 text-xs font-mono">
                                  {r.failures.map((f, fIdx) => (
                                    <div key={fIdx} className="p-3 bg-[#0e0b14] rounded-xl border border-white/5">
                                      <div className="text-slate-400 font-bold mb-1">Q: {f.question}</div>
                                      <div className="text-emerald-400 font-semibold">Expected: {f.expected}</div>
                                      <div className="text-[#ff8a65] font-semibold">Generated: {f.generated || '(blank)'}</div>
                                    </div>
                                  ))}
                                </div>
                              </div>
                            </td>
                          </tr>
                        )}
                      </React.Fragment>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* 2. Real-Time Toxic Token Suppression */}
      <div className="hardware-chassis p-6 rounded-3xl space-y-5">
        <div>
          <h3 className="text-sm font-bold text-white flex items-center gap-2">
            <ShieldCheck className="w-4 h-4 text-emerald-400" />
            Real-Time Toxic-Token Logit Suppression (-inf Masking)
          </h3>
          <p className="text-xs text-slate-400">
            Pre-computed single-token and multi-token blocklist sets logits to -&infin; before softmax normalization.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-12 gap-4">
          <div className="md:col-span-6 hardware-recess p-3 rounded-2xl">
            <label className="block text-[10px] font-bold uppercase tracking-wider text-slate-400 mb-1">
              Provocative Prompt
            </label>
            <input
              type="text"
              value={toxicPrompt}
              onChange={(e) => setToxicPrompt(e.target.value)}
              className="w-full bg-[#14101c] border border-white/10 px-3 py-1.5 rounded-xl text-xs text-slate-200 font-mono focus:outline-none focus:border-fuchsia-500"
            />
          </div>
          <div className="md:col-span-2 hardware-recess p-3 rounded-2xl">
            <label className="block text-[10px] font-bold uppercase tracking-wider text-slate-400 mb-1">
              Temp (T)
            </label>
            <input
              type="number"
              step="0.05"
              value={toxicTemp}
              onChange={(e) => setToxicTemp(parseFloat(e.target.value) || 1.0)}
              className="w-full bg-[#14101c] border border-white/10 px-2 py-1.5 rounded-xl text-xs font-mono text-center text-slate-200"
            />
          </div>
          <div className="md:col-span-2 hardware-recess p-3 rounded-2xl">
            <label className="block text-[10px] font-bold uppercase tracking-wider text-slate-400 mb-1">
              Seed
            </label>
            <input
              type="number"
              value={toxicSeed}
              onChange={(e) => setToxicSeed(parseInt(e.target.value) || 0)}
              className="w-full bg-[#14101c] border border-white/10 px-2 py-1.5 rounded-xl text-xs font-mono text-center text-slate-200"
            />
          </div>
          <div className="md:col-span-2 flex items-end">
            <button
              onClick={handleTestToxic}
              disabled={toxicLoading}
              className="btn-pill-magenta w-full text-white py-2.5 rounded-full text-xs font-bold tracking-wide flex items-center justify-center gap-1.5 cursor-pointer"
            >
              {toxicLoading ? <RefreshCw className="w-3.5 h-3.5 animate-spin" /> : <ShieldCheck className="w-3.5 h-3.5" />}
              <span>TEST FILTER</span>
            </button>
          </div>
        </div>

        {toxicData && (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 pt-1">
            <div className="hardware-recess p-4 rounded-2xl border border-rose-500/20">
              <span className="text-xs font-bold text-rose-400 uppercase tracking-wide block mb-2">
                Unfiltered Generation (Raw GPT-2)
              </span>
              <p className="text-xs font-mono text-slate-200 leading-relaxed bg-[#14101c] p-3 rounded-xl border border-white/5">
                {toxicData.unfiltered}
              </p>
            </div>
            <div className="hardware-recess p-4 rounded-2xl border border-emerald-500/20">
              <div className="flex justify-between items-center mb-2">
                <span className="text-xs font-bold text-emerald-400 uppercase tracking-wide">
                  Filtered (Suppression Active)
                </span>
                <span className="text-[10px] font-mono font-bold text-emerald-400 bg-emerald-950/60 px-2 py-0.5 rounded-md border border-emerald-700/50">
                  Peak mass removed: {(toxicData.peak_suppressed_mass * 100).toFixed(2)}%
                </span>
              </div>
              <p className="text-xs font-mono text-slate-200 leading-relaxed bg-[#14101c] p-3 rounded-xl border border-white/5">
                {toxicData.filtered}
              </p>
            </div>
          </div>
        )}
      </div>

      {/* 3. Token Probability Calibration & Model Card */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        
        {/* Calibration */}
        <div className="lg:col-span-7 hardware-chassis p-6 rounded-3xl space-y-4">
          <div className="flex justify-between items-center border-b border-white/5 pb-3">
            <div>
              <h3 className="text-sm font-bold text-white flex items-center gap-2">
                <Scale className="w-4 h-4 text-violet-400" />
                Token Calibration (Reliability Diagram)
              </h3>
              <p className="text-[11px] text-slate-400">
                Measures top-1 probability confidence vs empirical accuracy on WikiText-2.
              </p>
            </div>
            <button
              onClick={handleComputeCalibration}
              disabled={calibLoading}
              className="btn-pill-magenta text-white px-4 py-2 rounded-full text-xs font-bold tracking-wide flex items-center gap-1.5 cursor-pointer"
            >
              {calibLoading ? <RefreshCw className="w-3.5 h-3.5 animate-spin" /> : <Scale className="w-3.5 h-3.5" />}
              <span>CALIBRATE</span>
            </button>
          </div>

          {calibData ? (
            <div className="space-y-4">
              <div className="hardware-recess p-4 rounded-2xl flex flex-col justify-between">
                {/* SVG Calibration Chart with explicit axes and grid */}
                <div className="relative w-full h-56">
                  <svg className="w-full h-full select-none" viewBox="0 0 340 200">
                    <defs>
                      <filter id="neonBlur" x="-20%" y="-20%" width="140%" height="140%">
                        <feDropShadow dx="0" dy="0" stdDeviation="2.5" floodColor="#d946ef" floodOpacity="0.6" />
                      </filter>
                    </defs>

                    {/* Horizontal grid lines and Y-axis labels */}
                    {[0, 0.25, 0.5, 0.75, 1.0].map((tick) => {
                      const y = 165 - tick * 150;
                      return (
                        <g key={tick}>
                          <line x1="42" y1={y} x2="325" y2={y} stroke="rgba(255,255,255,0.06)" strokeDasharray="2 3" />
                          <text x="35" y={y + 3} textAnchor="end" fill="#64748b" fontSize="9" fontFamily="monospace">
                            {Math.round(tick * 100)}%
                          </text>
                        </g>
                      );
                    })}

                    {/* Vertical grid lines and X-axis labels */}
                    {[0, 0.2, 0.4, 0.6, 0.8, 1.0].map((tick) => {
                      const x = 42 + tick * 283;
                      return (
                        <g key={tick}>
                          <line x1={x} y1="15" x2={x} y2="165" stroke="rgba(255,255,255,0.06)" strokeDasharray="2 3" />
                          <text x={x} y="180" textAnchor="middle" fill="#64748b" fontSize="9" fontFamily="monospace">
                            {tick.toFixed(1)}
                          </text>
                        </g>
                      );
                    })}

                    {/* Ideal calibration diagonal y = x */}
                    <line x1="42" y1="165" x2="325" y2="15" stroke="#94a3b8" strokeWidth="1.5" strokeDasharray="4 4" opacity="0.6" />

                    {/* Polyline and interactive points for bins */}
                    {(() => {
                      const validBins: { i: number; cx: number; cy: number; conf: number; acc: number; count: number }[] = [];
                      calibData.bin_conf.forEach((conf, i) => {
                        const count = calibData.bin_count[i];
                        if (count > 0) {
                          const cx = 42 + Math.min(Math.max(conf, 0), 1) * 283;
                          const cy = 165 - Math.min(Math.max(calibData.bin_acc[i], 0), 1) * 150;
                          validBins.push({ i, cx, cy, conf, acc: calibData.bin_acc[i], count });
                        }
                      });

                      const pointsStr = validBins.map((b) => `${b.cx.toFixed(1)},${b.cy.toFixed(1)}`).join(" ");

                      return (
                        <>
                          <polyline fill="none" stroke="#d946ef" strokeWidth="2.5" points={pointsStr} filter="url(#neonBlur)" />

                          {validBins.map((b) => {
                            const isHovered = hoveredCalibBin?.bin === b.i;
                            return (
                              <g
                                key={b.i}
                                className="cursor-pointer"
                                onMouseEnter={() => setHoveredCalibBin({ bin: b.i, conf: b.conf, acc: b.acc, count: b.count })}
                                onMouseLeave={() => setHoveredCalibBin(null)}
                              >
                                <circle
                                  cx={b.cx}
                                  cy={b.cy}
                                  r={isHovered ? 6 : 4}
                                  fill={isHovered ? "#ffffff" : "#d946ef"}
                                  stroke="#0e0b14"
                                  strokeWidth="2"
                                  className="transition-all duration-150"
                                />
                              </g>
                            );
                          })}
                        </>
                      );
                    })()}
                  </svg>
                </div>

                {/* Legend and Axis Descriptors */}
                <div className="flex items-center justify-between text-[11px] font-mono text-slate-400 border-t border-white/5 pt-2 mt-1">
                  <div className="flex items-center gap-3">
                    <span className="flex items-center gap-1.5">
                      <span className="w-3 h-0.5 bg-slate-400 border-b border-dashed inline-block" />
                      <span>Ideal (y = x)</span>
                    </span>
                    <span className="flex items-center gap-1.5">
                      <span className="w-2.5 h-2.5 rounded-full bg-fuchsia-500 shadow-[0_0_6px_#d946ef] inline-block" />
                      <span className="text-fuchsia-300 font-bold">GPT-2 Empirical</span>
                    </span>
                  </div>
                  <span className="text-slate-500 text-[10px]">X: Top-1 Confidence &bull; Y: Empirical Accuracy</span>
                </div>

                {/* Interactive Tooltip Inspector */}
                <div className="mt-2 p-2.5 hardware-recess rounded-xl text-xs font-mono flex items-center justify-between">
                  {hoveredCalibBin ? (
                    <span className="text-slate-200">
                      Bin {hoveredCalibBin.bin + 1}: Conf <span className="text-violet-300 font-bold">{(hoveredCalibBin.conf * 100).toFixed(1)}%</span> &rarr; Acc <span className="text-fuchsia-400 font-bold">{(hoveredCalibBin.acc * 100).toFixed(1)}%</span> ({hoveredCalibBin.count} tokens)
                    </span>
                  ) : (
                    <span className="text-slate-500 italic text-[11px]">Hover over any circular bin marker to inspect exact values</span>
                  )}
                  {hoveredCalibBin && (
                    <span className={`text-[10px] font-bold px-2 py-0.5 rounded ${
                      hoveredCalibBin.acc < hoveredCalibBin.conf
                        ? "bg-amber-950/60 text-amber-300 border border-amber-800/40"
                        : "bg-emerald-950/60 text-emerald-300 border border-emerald-800/40"
                    }`}>
                      {hoveredCalibBin.acc < hoveredCalibBin.conf ? "Overconfident" : "Underconfident"}
                    </span>
                  )}
                </div>
              </div>

              <div className="flex items-center justify-between p-3.5 hardware-recess rounded-xl">
                <div>
                  <span className="text-xs text-slate-300 font-medium block">Expected Calibration Error (ECE)</span>
                  <span className="text-[10px] text-slate-500 font-mono">Weighted absolute gap across all 10 confidence bins</span>
                </div>
                <span className="text-lg font-mono font-extrabold text-fuchsia-400">
                  {calibData.ece.toFixed(4)}
                </span>
              </div>
            </div>
          ) : (
            <div className="hardware-recess p-8 rounded-2xl text-center">
              <Scale className="w-6 h-6 text-violet-500/40 mx-auto mb-2" />
              <span className="text-xs text-slate-400">
                Click "CALIBRATE" to compute 10-bin ECE over WikiText-2.
              </span>
            </div>
          )}
        </div>

        {/* Model Card & Governance */}
        <div className="lg:col-span-5 hardware-chassis p-6 rounded-3xl space-y-4 flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between border-b border-white/5 pb-3 mb-3">
              <h3 className="text-sm font-bold text-white flex items-center gap-2">
                <BookOpen className="w-4 h-4 text-slate-400" />
                Asset Licensing Summary
              </h3>
              <button
                onClick={handleLoadModelCard}
                className="text-xs text-fuchsia-400 hover:text-fuchsia-300 font-bold flex items-center gap-1"
              >
                <span>{showModelCard ? 'Hide' : 'Model Card'}</span>
                {showModelCard ? <ChevronUp className="w-3.5 h-3.5" /> : <ChevronDown className="w-3.5 h-3.5" />}
              </button>
            </div>

            <div className="hardware-recess p-2 rounded-2xl overflow-x-auto">
              <table className="w-full text-left text-xs border-collapse">
                <thead>
                  <tr className="border-b border-white/10 text-slate-400 font-bold">
                    <th className="py-2 px-2">Asset</th>
                    <th className="py-2 px-2">License</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-white/5 text-slate-300 font-mono text-[11px]">
                  <tr>
                    <td className="py-2 px-2 font-bold text-white">GPT-2 (124M)</td>
                    <td className="py-2 px-2 text-emerald-400">MIT</td>
                  </tr>
                  <tr>
                    <td className="py-2 px-2 font-bold text-white">BERT Base Uncased</td>
                    <td className="py-2 px-2 text-emerald-400">Apache-2.0</td>
                  </tr>
                  <tr>
                    <td className="py-2 px-2 font-bold text-white">sd-turbo</td>
                    <td className="py-2 px-2 text-[#ff8a65]">Stability Community</td>
                  </tr>
                  <tr>
                    <td className="py-2 px-2 font-bold text-white">WikiText-2</td>
                    <td className="py-2 px-2 text-violet-400">CC BY-SA 3.0</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>

          {showModelCard && modelCardContent && (
            <div className="mt-3 p-4 hardware-recess rounded-2xl text-[11px] font-mono text-slate-300 max-h-56 overflow-y-auto whitespace-pre-wrap leading-relaxed">
              {modelCardContent}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

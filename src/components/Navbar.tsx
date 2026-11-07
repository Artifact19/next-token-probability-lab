import React from 'react';
import { Sliders, Activity, Disc3 } from 'lucide-react';

interface NavbarProps {
  activeTab: number;
  setActiveTab: (tab: number) => void;
  systemHealth: { status: string; device: string; has_hf_token?: boolean } | null;
}

export const Navbar: React.FC<NavbarProps> = ({ activeTab, setActiveTab, systemHealth }) => {
  const tabs = [
    { id: 0, label: "U1 Sampling", code: "PROB.SAMPLER" },
    { id: 1, label: "U2 Attention", code: "ATTN.LSTM" },
    { id: 2, label: "U3 BERT vs GPT", code: "MASK.CAUSAL" },
    { id: 3, label: "U4 Sampler & SD", code: "KV.DIFFUSE" },
    { id: 4, label: "U5 Responsible AI", code: "AUDIT.SAFETY" },
  ];

  return (
    <header className="sticky top-0 z-50 px-3 sm:px-6 lg:px-8 pt-3 sm:pt-4 pb-2">
      <div className="max-w-7xl mx-auto hardware-chassis rounded-2xl sm:rounded-3xl p-3 sm:p-4 backdrop-blur-xl">
        <div className="flex flex-col lg:flex-row items-stretch lg:items-center justify-between gap-3 sm:gap-4">
          
          {/* Hardware Console Brand and Top Status Row on Mobile */}
          <div className="flex items-center justify-between gap-3">
            <div className="flex items-center gap-3">
              {/* Illuminated Glowing Status Knob Indicator */}
              <div className="w-9 h-9 sm:w-10 sm:h-10 rounded-xl sm:rounded-2xl bg-gradient-to-br from-[#2f2442] to-[#171322] shadow-[0_4px_12px_rgba(0,0,0,0.8),inset_0_1px_1px_rgba(255,255,255,0.2)] border border-white/10 flex items-center justify-center relative shrink-0">
                <div className="w-3.5 h-3.5 sm:w-4 sm:h-4 rounded-full bg-gradient-to-tr from-[#d946ef] to-[#ff8a65] shadow-[0_0_12px_#d946ef] animate-pulse" />
              </div>

              <div>
                <div className="flex items-center gap-2 flex-wrap">
                  <h1 className="font-extrabold text-sm sm:text-base tracking-tight text-white flex items-center gap-1.5">
                    NEXT-TOKEN
                    <span className="text-transparent bg-clip-text bg-gradient-to-r from-fuchsia-400 via-pink-400 to-amber-300">
                      LAB
                    </span>
                  </h1>
                  <span className="px-2 py-0.5 rounded-full text-[9px] sm:text-[10px] font-mono font-bold bg-[#261e33] border border-white/10 text-fuchsia-300 shadow-sm">
                    Track A &bull; Problem P5
                  </span>
                </div>
                <p className="text-[10px] sm:text-[11px] text-slate-400 font-medium line-clamp-1">
                  Interactive Language Model Probability &amp; Sampling Laboratory
                </p>
              </div>
            </div>

            {/* Mobile-visible compact status badges */}
            <div className="flex lg:hidden items-center gap-1.5 shrink-0">
              <div className="flex items-center gap-1 px-2 py-1 rounded-lg bg-[#120f18] border border-white/5 shadow-inner whitespace-nowrap">
                <span className={`w-1.5 h-1.5 rounded-full shrink-0 ${systemHealth?.has_hf_token ? "bg-emerald-400 shadow-[0_0_6px_#34d399]" : "bg-amber-400 shadow-[0_0_6px_#fbbf24]"}`} />
                <span className="text-[10px] font-mono text-slate-400">HF:</span>
                <span className={`text-[10px] font-mono font-bold ${systemHealth?.has_hf_token ? "text-emerald-300" : "text-amber-300"}`}>
                  {systemHealth?.has_hf_token ? "ON" : "OFF"}
                </span>
              </div>
              <div className="flex items-center gap-1 px-2 py-1 rounded-lg bg-[#120f18] border border-white/5 shadow-inner whitespace-nowrap">
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 shadow-[0_0_6px_#34d399] shrink-0" />
                <span className="text-[10px] font-mono font-bold text-slate-200">
                  {systemHealth?.device?.toUpperCase() || 'CPU'}
                </span>
              </div>
            </div>
          </div>

          {/* Navigation Tabs - Horizontally scrollable on mobile */}
          <div className="flex items-center gap-1 sm:gap-1.5 overflow-x-auto max-w-full p-1 hardware-recess rounded-xl sm:rounded-2xl scrollbar-none">
            {tabs.map((tab) => {
              const isActive = activeTab === tab.id;
              return (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={`px-2.5 sm:px-3 py-1 sm:py-1.5 rounded-lg sm:rounded-xl text-[11px] sm:text-xs font-semibold transition-all whitespace-nowrap flex items-center gap-1.5 shrink-0 cursor-pointer ${
                    isActive
                      ? "btn-pill-magenta text-white shadow-sm"
                      : "text-slate-400 hover:text-slate-200 hover:bg-[#1a1424]"
                  }`}
                >
                  {isActive && <span className="w-1.5 h-1.5 rounded-full bg-white shadow-[0_0_6px_#fff]" />}
                  <span>{tab.label}</span>
                </button>
              );
            })}
          </div>

          {/* Desktop Right Status LED Indicators */}
          <div className="hidden lg:flex items-center gap-2 shrink-0">
            <div className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-xl bg-[#120f18] border border-white/5 shadow-inner whitespace-nowrap">
              <span className={`w-2 h-2 rounded-full shrink-0 ${systemHealth?.has_hf_token ? "bg-emerald-400 shadow-[0_0_8px_#34d399]" : "bg-amber-400 shadow-[0_0_8px_#fbbf24]"}`} />
              <span className="text-[11px] font-mono text-slate-400">HF:</span>
              <span className={`text-[11px] font-mono font-bold ${systemHealth?.has_hf_token ? "text-emerald-300" : "text-amber-300"}`}>
                {systemHealth?.has_hf_token ? "CONNECTED" : "OFFLINE"}
              </span>
            </div>

            <div className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-xl bg-[#120f18] border border-white/5 shadow-inner whitespace-nowrap">
              <span className="w-2 h-2 rounded-full bg-emerald-400 shadow-[0_0_8px_#34d399] shrink-0" />
              <span className="text-[11px] font-mono text-slate-400">DEVICE:</span>
              <span className="text-[11px] font-mono font-bold text-slate-200">
                {systemHealth?.device?.toUpperCase() || 'CPU'}
              </span>
            </div>
          </div>
        </div>
      </div>
    </header>
  );
};

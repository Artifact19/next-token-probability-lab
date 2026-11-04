import React, { useRef, useState, useEffect } from 'react';

interface KnobProps {
  label: string;
  value: number;
  min: number;
  max: number;
  step?: number;
  unit?: string;
  color?: 'amber' | 'magenta' | 'violet';
  defaultValue?: number;
  onChange: (val: number) => void;
}

export const Knob: React.FC<KnobProps> = ({
  label,
  value,
  min,
  max,
  step = 0.05,
  unit = '',
  color = 'magenta',
  defaultValue,
  onChange,
}) => {
  const knobRef = useRef<HTMLDivElement>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [isEditing, setIsEditing] = useState(false);
  const [editValue, setEditValue] = useState(value.toString());
  const inputRef = useRef<HTMLInputElement>(null);

  const startY = useRef(0);
  const startX = useRef(0);
  const startVal = useRef(value);

  // Normalize angle between -135deg and +135deg (270deg sweep)
  const percent = Math.min(Math.max((value - min) / (max - min), 0), 1);
  const angle = -135 + percent * 270;

  // Sync editValue when external value changes and not actively typing
  useEffect(() => {
    if (!isEditing) {
      setEditValue(value.toFixed(step >= 1 ? 0 : 2));
    }
  }, [value, isEditing, step]);

  useEffect(() => {
    if (isEditing && inputRef.current) {
      inputRef.current.focus();
      inputRef.current.select();
    }
  }, [isEditing]);

  const handleNudge = (direction: -1 | 1) => {
    const nudgeStep = step || (max - min) / 20;
    let newVal = Math.min(Math.max(value + direction * nudgeStep, min), max);
    if (step) {
      newVal = Math.round(newVal / step) * step;
    }
    onChange(parseFloat(newVal.toFixed(2)));
  };

  const handleEditSubmit = () => {
    setIsEditing(false);
    const parsed = parseFloat(editValue);
    if (!isNaN(parsed)) {
      let clamped = Math.min(Math.max(parsed, min), max);
      if (step) {
        clamped = Math.round(clamped / step) * step;
      }
      onChange(parseFloat(clamped.toFixed(2)));
    } else {
      setEditValue(value.toFixed(step >= 1 ? 0 : 2));
    }
  };

  const handleMouseDown = (e: React.MouseEvent) => {
    if ((e.target as HTMLElement).tagName === 'BUTTON' || (e.target as HTMLElement).tagName === 'INPUT') {
      return;
    }
    setIsDragging(true);
    startY.current = e.clientY;
    startX.current = e.clientX;
    startVal.current = value;

    const handleMouseMove = (moveEvent: MouseEvent) => {
      // Holding Shift enables high-precision fine adjustment
      const sensitivity = moveEvent.shiftKey ? 280 : 70;
      // Moving up or right increases value, moving down or left decreases value
      const deltaY = startY.current - moveEvent.clientY;
      const deltaX = moveEvent.clientX - startX.current;
      const delta = deltaY + deltaX * 0.6;

      const range = max - min;
      const change = (delta / sensitivity) * range;
      let newVal = Math.min(Math.max(startVal.current + change, min), max);
      if (step) {
        newVal = Math.round(newVal / step) * step;
      }
      onChange(parseFloat(newVal.toFixed(2)));
    };

    const handleMouseUp = () => {
      setIsDragging(false);
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('mouseup', handleMouseUp);
    };

    window.addEventListener('mousemove', handleMouseMove);
    window.addEventListener('mouseup', handleMouseUp);
  };

  const handleTouchStart = (e: React.TouchEvent) => {
    if (e.touches.length !== 1) return;
    if ((e.target as HTMLElement).tagName === 'BUTTON' || (e.target as HTMLElement).tagName === 'INPUT') {
      return;
    }
    setIsDragging(true);
    startY.current = e.touches[0].clientY;
    startX.current = e.touches[0].clientX;
    startVal.current = value;

    const handleTouchMove = (moveEvent: TouchEvent) => {
      if (moveEvent.touches.length !== 1) return;
      moveEvent.preventDefault();
      const deltaY = startY.current - moveEvent.touches[0].clientY;
      const deltaX = moveEvent.touches[0].clientX - startX.current;
      const delta = deltaY + deltaX * 0.6;

      const range = max - min;
      const change = (delta / 70) * range;
      let newVal = Math.min(Math.max(startVal.current + change, min), max);
      if (step) {
        newVal = Math.round(newVal / step) * step;
      }
      onChange(parseFloat(newVal.toFixed(2)));
    };

    const handleTouchEnd = () => {
      setIsDragging(false);
      window.removeEventListener('touchmove', handleTouchMove);
      window.removeEventListener('touchend', handleTouchEnd);
      window.removeEventListener('touchcancel', handleTouchEnd);
    };

    window.addEventListener('touchmove', handleTouchMove, { passive: false });
    window.addEventListener('touchend', handleTouchEnd);
    window.addEventListener('touchcancel', handleTouchEnd);
  };

  const handleWheel = (e: React.WheelEvent) => {
    e.preventDefault();
    const delta = -Math.sign(e.deltaY);
    const stepVal = step || (max - min) / 30;
    let newVal = Math.min(Math.max(value + delta * stepVal, min), max);
    if (step) {
      newVal = Math.round(newVal / step) * step;
    }
    onChange(parseFloat(newVal.toFixed(2)));
  };

  const handleDoubleClick = () => {
    const target = defaultValue !== undefined ? defaultValue : (min + max) / 2;
    let clamped = Math.min(Math.max(target, min), max);
    if (step) {
      clamped = Math.round(clamped / step) * step;
    }
    onChange(parseFloat(clamped.toFixed(2)));
  };

  const handleArcClick = (e: React.MouseEvent) => {
    const rect = knobRef.current?.getBoundingClientRect();
    if (!rect) return;
    const cx = rect.left + rect.width / 2;
    const cy = rect.top + rect.height / 2;
    const deg = Math.atan2(e.clientY - cy, e.clientX - cx) * (180 / Math.PI);
    let knobDeg = deg + 90;
    if (knobDeg > 180) knobDeg -= 360;

    if (knobDeg >= -145 && knobDeg <= 145) {
      const clampedDeg = Math.min(Math.max(knobDeg, -135), 135);
      const newPercent = (clampedDeg + 135) / 270;
      let newVal = min + newPercent * (max - min);
      if (step) {
        newVal = Math.round(newVal / step) * step;
      }
      onChange(parseFloat(newVal.toFixed(2)));
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (isEditing) return;
    const largeStep = (step || 1) * 5;

    if (e.key === 'ArrowUp' || e.key === 'ArrowRight') {
      e.preventDefault();
      handleNudge(1);
    } else if (e.key === 'ArrowDown' || e.key === 'ArrowLeft') {
      e.preventDefault();
      handleNudge(-1);
    } else if (e.key === 'PageUp') {
      e.preventDefault();
      let newVal = Math.min(value + largeStep, max);
      if (step) newVal = Math.round(newVal / step) * step;
      onChange(parseFloat(newVal.toFixed(2)));
    } else if (e.key === 'PageDown') {
      e.preventDefault();
      let newVal = Math.max(value - largeStep, min);
      if (step) newVal = Math.round(newVal / step) * step;
      onChange(parseFloat(newVal.toFixed(2)));
    } else if (e.key === 'Home') {
      e.preventDefault();
      onChange(min);
    } else if (e.key === 'End') {
      e.preventDefault();
      onChange(max);
    } else if (e.key === 'Enter') {
      e.preventDefault();
      setIsEditing(true);
    }
  };

  // SVG circular arc parameters (r = 29, circumference = 2 * PI * 29 = 182.21)
  const radius = 29;
  const circumference = 2 * Math.PI * radius; // ~182.21
  const arcLength = circumference * (270 / 360); // ~136.66
  const dashOffset = arcLength * (1 - percent);

  const strokeColor =
    color === 'amber' ? '#ff8a65' :
    color === 'violet' ? '#a855f7' :
    '#d946ef';

  const glowClass = 
    color === 'amber' ? 'shadow-[0_0_20px_rgba(255,138,101,0.5)] border-[#ff8a65]' :
    color === 'violet' ? 'shadow-[0_0_20px_rgba(139,92,246,0.5)] border-[#a855f7]' :
    'shadow-[0_0_20px_rgba(217,70,239,0.55)] border-[#d946ef]';

  return (
    <div
      tabIndex={0}
      onKeyDown={handleKeyDown}
      className="flex flex-col items-center select-none group focus:outline-none"
      title={`${label}: ${value} (Drag, scroll wheel, click arc, or use +/-)`}
    >
      {/* Outer Dial Cavity with SVG Glowing Arc */}
      <div
        ref={knobRef}
        onMouseDown={handleMouseDown}
        onTouchStart={handleTouchStart}
        onWheel={handleWheel}
        onDoubleClick={handleDoubleClick}
        onClick={handleArcClick}
        className="w-18 h-18 sm:w-20 sm:h-20 rounded-full p-1.5 bg-[#14101c] shadow-[inset_0_3px_8px_rgba(0,0,0,0.8),0_1px_0_rgba(255,255,255,0.08)] flex items-center justify-center cursor-pointer relative transition-transform touch-none"
      >
        {/* Circular SVG Progress Track */}
        <svg
          className="absolute inset-0 w-full h-full pointer-events-none"
          viewBox="0 0 72 72"
        >
          {/* Background Inactive Arc (270 deg, starts at 135 deg / 7:30) */}
          <circle
            cx="36"
            cy="36"
            r={radius}
            fill="none"
            stroke="rgba(255, 255, 255, 0.08)"
            strokeWidth="3.5"
            strokeDasharray={`${arcLength} ${circumference}`}
            strokeLinecap="round"
            transform="rotate(135 36 36)"
          />
          {/* Active Illuminated Neon Arc */}
          <circle
            cx="36"
            cy="36"
            r={radius}
            fill="none"
            stroke={strokeColor}
            strokeWidth="3.5"
            strokeDasharray={`${arcLength} ${circumference}`}
            strokeDashoffset={dashOffset}
            strokeLinecap="round"
            transform="rotate(135 36 36)"
            style={{
              filter: `drop-shadow(0 0 4px ${strokeColor})`,
              transition: isDragging ? 'none' : 'stroke-dashoffset 0.15s ease',
            }}
          />
        </svg>

        {/* Rotary Knob Cylinder */}
        <div
          className={`w-12 h-12 sm:w-13 sm:h-13 rounded-full knob-housing relative flex items-center justify-center transition-all cursor-grab active:cursor-grabbing ${
            isDragging ? `${glowClass} scale-105` : 'hover:shadow-[0_0_15px_rgba(217,70,239,0.35)]'
          }`}
          style={{
            transform: `rotate(${angle}deg)`,
          }}
        >
          {/* Fluted Edge / Knurling Pattern */}
          <div className="absolute inset-0 rounded-full knob-knurled opacity-40" />

          {/* Top Metallic Cap */}
          <div className="w-8 h-8 sm:w-9 sm:h-9 rounded-full bg-gradient-to-b from-[#2e263c] to-[#1c1726] shadow-[inset_0_1px_1px_rgba(255,255,255,0.25),0_3px_6px_rgba(0,0,0,0.6)] flex items-start justify-center pt-1 z-10">
            {/* Indicator Dot / Notch */}
            <div
              className={`w-1.5 h-1.5 sm:w-1.5 sm:h-2 rounded-full ${
                color === 'amber'
                  ? 'bg-[#ff8a65] shadow-[0_0_8px_#ff8a65]'
                  : color === 'violet'
                  ? 'bg-[#a855f7] shadow-[0_0_8px_#a855f7]'
                  : 'bg-[#e879f9] shadow-[0_0_8px_#d946ef]'
              }`}
            />
          </div>
        </div>
      </div>

      {/* Label */}
      <span className="text-[10px] sm:text-[11px] font-semibold tracking-wider uppercase text-slate-400 mt-2">
        {label}
      </span>

      {/* Stepper Controls & Value Display */}
      <div className="flex items-center gap-1.5 mt-1">
        {/* Nudge Down Button */}
        <button
          type="button"
          onClick={() => handleNudge(-1)}
          className="w-5 h-5 rounded-md bg-[#161220] hover:bg-[#281f37] active:scale-90 border border-white/10 text-slate-400 hover:text-white flex items-center justify-center text-xs font-bold transition-all select-none shadow-sm cursor-pointer"
          title="Decrease"
        >
          -
        </button>

        {/* Editable Value Badge */}
        {isEditing ? (
          <input
            ref={inputRef}
            type="number"
            value={editValue}
            step={step}
            min={min}
            max={max}
            onChange={(e) => setEditValue(e.target.value)}
            onBlur={handleEditSubmit}
            onKeyDown={(e) => {
              if (e.key === 'Enter') handleEditSubmit();
              if (e.key === 'Escape') {
                setIsEditing(false);
                setEditValue(value.toFixed(step >= 1 ? 0 : 2));
              }
            }}
            className="w-16 text-center text-xs font-mono font-bold text-white bg-[#0e0b14] px-1 py-0.5 rounded-md border border-fuchsia-500 outline-none shadow-[0_0_8px_rgba(217,70,239,0.5)]"
          />
        ) : (
          <button
            type="button"
            onClick={() => setIsEditing(true)}
            className="text-xs font-mono font-bold text-white bg-[#0e0b14] hover:bg-[#1a1427] hover:border-white/20 px-2 py-0.5 rounded-md border border-white/5 transition-all cursor-text"
            title="Click to type exact number"
          >
            {value.toFixed(step >= 1 ? 0 : 2)}{unit}
          </button>
        )}

        {/* Nudge Up Button */}
        <button
          type="button"
          onClick={() => handleNudge(1)}
          className="w-5 h-5 rounded-md bg-[#161220] hover:bg-[#281f37] active:scale-90 border border-white/10 text-slate-400 hover:text-white flex items-center justify-center text-xs font-bold transition-all select-none shadow-sm cursor-pointer"
          title="Increase"
        >
          +
        </button>
      </div>
    </div>
  );
};

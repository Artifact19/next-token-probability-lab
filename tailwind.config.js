/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  darkMode: 'class',
  theme: {
    extend: {
      fontFamily: {
        sans: ['Outfit', 'sans-serif'],
        mono: ['JetBrains Mono', 'monospace'],
      },
      colors: {
        synth: {
          bg: '#110e17',
          chassis: '#191522',
          surface: '#221c2e',
          recess: '#0d0a13',
          border: 'rgba(255, 255, 255, 0.07)',
          knob: '#2a2438',
          accent: {
            magenta: '#d946ef',
            pink: '#f43f5e',
            violet: '#8b5cf6',
            amber: '#ff8a65',
            peach: '#fb923c',
            cyan: '#38bdf8',
          }
        }
      },
      boxShadow: {
        'chassis': '0 30px 80px -15px rgba(0,0,0,0.85), inset 0 1px 1px 0 rgba(255,255,255,0.08), inset 0 -1px 2px 0 rgba(0,0,0,0.6)',
        'knob': '0 8px 20px -3px rgba(0,0,0,0.8), inset 0 1px 1px rgba(255,255,255,0.2), inset 0 -3px 6px rgba(0,0,0,0.7)',
        'knob-glow': '0 0 25px 4px rgba(255, 138, 101, 0.35), 0 8px 20px -3px rgba(0,0,0,0.8)',
        'pill-magenta': '0 12px 30px -4px rgba(217, 70, 239, 0.55), inset 0 1px 1px rgba(255, 255, 255, 0.4), inset 0 -2px 4px rgba(0,0,0,0.3)',
        'pill-dark': '0 6px 16px -2px rgba(0,0,0,0.6), inset 0 1px 1px rgba(255, 255, 255, 0.12), inset 0 -2px 4px rgba(0,0,0,0.5)',
        'recessed': 'inset 0 3px 8px rgba(0,0,0,0.7), inset 0 1px 2px rgba(0,0,0,0.9), 0 1px 0 rgba(255,255,255,0.06)',
        'glow-bar': '0 0 15px rgba(217, 70, 239, 0.8), 0 0 30px rgba(217, 70, 239, 0.4)',
      }
    },
  },
  plugins: [],
}

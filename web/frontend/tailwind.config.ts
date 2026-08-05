import type { Config } from 'tailwindcss';

const config: Config = {
  content: [
    './pages/**/*.{js,ts,jsx,tsx,mdx}',
    './components/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        // KAAL brand
        'kaal-red':   '#CC0000',
        'kaal-dark':  '#0A0A0A',
        'kaal-panel': '#111111',
        'kaal-border':'#222222',
        // Risk tier colours
        'risk-robust':      '#4ADE80',
        'risk-low':         '#A3E635',
        'risk-medium':      '#FACC15',
        'risk-high':        '#FB923C',
        'risk-critical':    '#CC0000',
        'risk-catastrophic':'#7F0000',
      },
      fontFamily: {
        mono: ['JetBrains Mono', 'Fira Code', 'Courier New', 'monospace'],
      },
    },
  },
  plugins: [],
};

export default config;

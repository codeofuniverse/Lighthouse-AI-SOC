/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      colors: {
        base:      'var(--bg-base)',
        surface:   'var(--bg-surface)',
        panel:     'var(--bg-panel)',
        elevated:  'var(--bg-elevated)',
        border:    'var(--border)',
        critical:  'var(--critical)',
        suspicious:'var(--suspicious)',
        safe:      'var(--safe)',
        success:   'var(--success)',
        muted:     'var(--muted)',
      },
      fontFamily: {
        mono:    ['IBM Plex Mono', 'JetBrains Mono', 'Consolas', 'monospace'],
        display: ['Space Grotesk', 'system-ui', 'sans-serif'],
        sans:    ['Space Grotesk', 'system-ui', 'sans-serif'],
      },
    },
  },
  plugins: [],
}

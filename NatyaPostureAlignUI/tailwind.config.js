/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './pages/**/*.{js,ts,jsx,tsx,mdx}',
    './components/**/*.{js,ts,jsx,tsx,mdx}',
    './app/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      fontFamily: {
        display: ['var(--font-display)'],
        body: ['var(--font-body)'],
      },
      colors: {
        saffron:  { DEFAULT: '#E8621A', light: '#F4894E', dark: '#BF4A0E' },
        gold:     { DEFAULT: '#C8952A', light: '#E8B84B', dark: '#9E720D' },
        crimson:  { DEFAULT: '#8B1A2F', light: '#B8243F', dark: '#5C0F1E' },
        ivory:    { DEFAULT: '#FAF3E8', dark: '#EDE0C8' },
        ink:      { DEFAULT: '#1C1108', mid: '#3D2B14', light: '#6B4C2A' },
      },
      animation: {
        'fade-up':    'fadeUp 0.6s ease forwards',
        'fade-in':    'fadeIn 0.4s ease forwards',
        'pulse-slow': 'pulse 3s ease-in-out infinite',
        'spin-slow':  'spin 8s linear infinite',
      },
      keyframes: {
        fadeUp:  { '0%': { opacity: 0, transform: 'translateY(20px)' }, '100%': { opacity: 1, transform: 'translateY(0)' } },
        fadeIn:  { '0%': { opacity: 0 }, '100%': { opacity: 1 } },
      },
    },
  },
  plugins: [],
}
/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      colors: {
        ink: {
          950: "#0B1526", // page background
          900: "#101F38", // card / surface background
          800: "#182A47", // elevated surface, hover
          700: "#24395C", // borders, dividers
          600: "#33507D", // stronger borders, disabled states
        },
        parchment: {
          100: "#F4EFE3", // primary text
          300: "#C9C2AE", // muted text / secondary
          500: "#8D8672", // faint text, placeholders
        },
        gold: {
          300: "#E8C468", // hover / lighter accent
          500: "#C9A227", // primary accent — buttons, links, active states
          700: "#8C7016", // pressed state, borders on gold elements
        },
      },
      fontFamily: {
        display: ["Fraunces", "Georgia", "serif"],
        sans: ["IBM Plex Sans", "system-ui", "sans-serif"],
        mono: ["IBM Plex Mono", "ui-monospace", "monospace"],
      },
      borderRadius: {
        panel: "6px",
      },
      maxWidth: {
        column: "68ch",
      },
    },
  },
  plugins: [],
};

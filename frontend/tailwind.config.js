/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      colors: {
        sage: {
          950: "#22302A", // text primary
          900: "#33473B", // primary dark (headers, nav, primary buttons)
          700: "#4F7A63", // primary (links, active states, chart series 1)
          500: "#7FA98F", // secondary (hover states)
          200: "#DCE5DD", // neutral tint (card backgrounds, dividers)
          100: "#F8FAF7", // page background
          muted: "#5C6B62", // muted text
        },
        accent: {
          sand: "#C9A876" // used ONLY for negative-sentiment / warning states
        }
      },
      fontFamily: {
        sans: ["Inter", "system-ui", "sans-serif"]
      },
      borderRadius: {
        card: "12px"
      },
      boxShadow: {
        card: "0 1px 3px rgba(34, 48, 42, 0.08), 0 1px 2px rgba(34, 48, 42, 0.06)"
      }
    }
  },
  plugins: []
};

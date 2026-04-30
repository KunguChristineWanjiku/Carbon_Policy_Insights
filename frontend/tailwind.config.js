/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      colors: {
        bg: "#F4F6F9",
        panel: "#FFFFFF",
        accent: "#1A56DB",
        accent2: "#0E9F6E",
        danger: "#E02424",
        warn: "#FF8A4C",
        textc: "#111928",
        muted: "#6B7280",
      },
    },
  },
  plugins: [],
};

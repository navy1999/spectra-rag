import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        // Warm off-white "paper" palette — softer on the eyes than cool white.
        canvas: "#FBFAF5",
        offwhite: "#FBFAF5",
        sidebar: "#F4F2EB",
        panel: "#FFFEFA",
        accent: "#1C1B17",
        muted: "#76726A",
        border: "#E9E5DB",
        faint: "#F1EEE6",
        // Status semantics.
        live: "#10B981",
        warn: "#F59E0B",
        down: "#EF4444",
        // Regime semantics (flat, no gradients).
        logic: "#0EA5E9",
        creative: "#8B5CF6",
      },
      fontFamily: {
        sans: ["var(--font-inter)", "Inter", "system-ui", "sans-serif"],
        mono: [
          "var(--font-mono)",
          "ui-monospace",
          "SFMono-Regular",
          "Menlo",
          "Consolas",
          "monospace",
        ],
      },
    },
  },
  plugins: [],
};
export default config;

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
        // Cool, neutral instrument palette (Vercel/Linear-adjacent).
        canvas: "#FAFAFA",
        offwhite: "#FAFAFA",
        sidebar: "#F4F4F5",
        panel: "#FFFFFF",
        accent: "#18181B",
        muted: "#71717A",
        border: "#E4E4E7",
        faint: "#F4F4F5",
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

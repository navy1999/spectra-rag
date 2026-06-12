// next.config.mjs

/** @type {import('next').NextConfig} */
const nextConfig = {
  output: "standalone",
  env: {
    BACKEND_URL: process.env.BACKEND_URL ?? "http://localhost:8080",
  },
};

export default nextConfig;
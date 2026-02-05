import type { NextConfig } from "next";

const apiHost = process.env.IARA_SERVER_HOST ?? "0.0.0.0";
const apiPort = process.env.IARA_SERVER_PORT ?? "7860";

const nextConfig: NextConfig = {
  async rewrites() {
    return [
      {
        source: "/api/:path*",
        destination: `http://${apiHost}:${apiPort}/api/:path*`,
      },
    ];
  },
};

export default nextConfig;

/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  // 若設定 USE_LOCAL_FASTAPI=true 則將 API 請求轉送至本機 FastAPI 8000 埠
  // 在 Vercel 生產環境或標準部署時，使用同源 Next.js App Router API Route Handlers
  async rewrites() {
    if (process.env.USE_LOCAL_FASTAPI === 'true') {
      return [
        {
          source: '/api/:path*',
          destination: 'http://127.0.0.1:8000/api/:path*',
        },
      ];
    }
    return [];
  },
};

module.exports = nextConfig;

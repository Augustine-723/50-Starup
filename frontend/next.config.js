/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  // 支援本機開發代理 FastAPI 後端
  async rewrites() {
    return [
      {
        source: '/api/:path*',
        destination: 'http://127.0.0.1:8000/api/:path*',
      },
    ]
  },
}

module.exports = nextConfig

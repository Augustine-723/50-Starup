import { NextResponse } from 'next/server';

export const dynamic = 'force-dynamic';

export async function GET() {
  return NextResponse.json({
    status: 'ok',
    service: '50 Startups Multiple Linear Regression Explorer API',
    environment: process.env.NODE_ENV || 'production',
    vercel: Boolean(process.env.VERCEL),
    timestamp: new Date().toISOString(),
    endpoints: [
      'GET /api/results?random_state=42',
      'GET /api/models?random_state=42',
      'GET /api/optuna',
      'POST /api/predict',
      'GET /api/health',
    ],
  });
}

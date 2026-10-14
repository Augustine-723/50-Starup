import { NextResponse } from 'next/server';

export const dynamic = 'force-dynamic';

export async function GET() {
  const isExternalFastApi = Boolean(process.env.NEXT_PUBLIC_API_URL);

  return NextResponse.json({
    status: 'ok',
    service: '50 Startups Multiple Linear Regression Explorer API',
    environment: process.env.NODE_ENV || 'production',
    vercel: Boolean(process.env.VERCEL),
    runtime: isExternalFastApi ? 'FastAPI Proxy' : 'Vercel Serverless (Next.js App Router)',
    backend_mode: isExternalFastApi ? 'external_fastapi' : 'serverless_ml_api',
    model_engine: 'scikit-learn & statsmodels (Trained Pipeline Parameters)',
    timestamp: new Date().toISOString(),
    endpoints: [
      'GET /api/results?random_state=42',
      'GET /api/results?random_state=0',
      'GET /api/models?random_state=42',
      'GET /api/optuna',
      'POST /api/predict',
      'GET /api/health',
    ],
  });
}

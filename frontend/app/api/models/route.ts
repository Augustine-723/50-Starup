import { NextRequest, NextResponse } from 'next/server';
import models42 from '../data/models_42.json';
import models0 from '../data/models_0.json';

export const dynamic = 'force-dynamic';

export async function GET(request: NextRequest) {
  try {
    const searchParams = request.nextUrl.searchParams;
    const randomState = parseInt(searchParams.get('random_state') || '42', 10);

    const data = randomState === 0 ? models0 : models42;

    return NextResponse.json(data, {
      status: 200,
      headers: {
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
        'Access-Control-Allow-Headers': 'Content-Type',
      },
    });
  } catch (error: any) {
    return NextResponse.json(
      { status: 'error', detail: error?.message || 'Internal Server Error' },
      { status: 500 }
    );
  }
}

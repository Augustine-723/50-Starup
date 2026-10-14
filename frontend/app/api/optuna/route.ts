import { NextRequest, NextResponse } from 'next/server';
import optunaData from '../data/optuna_results.json';

export const dynamic = 'force-dynamic';

export async function GET(request: NextRequest) {
  try {
    return NextResponse.json(optunaData, {
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

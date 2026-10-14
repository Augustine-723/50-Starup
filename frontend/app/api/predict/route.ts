import { NextRequest, NextResponse } from 'next/server';
import results42 from '../data/results_42.json';
import results0 from '../data/results_0.json';

export const dynamic = 'force-dynamic';

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const {
      rd_spend = 0,
      administration = 0,
      marketing_spend = 0,
      state = 'New York',
      algorithm = 'Forward SFS',
      k = 3,
      random_state = 42,
    } = body;

    const validStates = ['New York', 'California', 'Florida'];
    if (!validStates.includes(state)) {
      return NextResponse.json(
        {
          status: 'error',
          detail: `Invalid state '${state}'. Allowed states: ${validStates.join(', ')}`,
        },
        { status: 400 }
      );
    }

    const dataSource = random_state === 0 ? results0 : results42;
    const fsResults = (dataSource as any).feature_selection_results;

    const algoModels = fsResults[algorithm] || fsResults['Forward SFS'] || [];
    const targetModel =
      algoModels.find((m: any) => m.k === Number(k)) || algoModels[2] || algoModels[0];

    if (!targetModel) {
      return NextResponse.json(
        { status: 'error', detail: 'Target model configuration not found.' },
        { status: 404 }
      );
    }

    // One-Hot Encoding values (California is baseline: Florida=0, NY=0)
    const stateFlorida = state === 'Florida' ? 1.0 : 0.0;
    const stateNewYork = state === 'New York' ? 1.0 : 0.0;

    const sample: Record<string, number> = {
      'R&D Spend': Number(rd_spend),
      'Administration': Number(administration),
      'Marketing Spend': Number(marketing_spend),
      'State_Florida': stateFlorida,
      'State_New York': stateNewYork,
    };

    let predicted = targetModel.intercept;
    const breakdown: Record<string, number> = {
      Intercept: targetModel.intercept,
    };

    for (const [feat, weight] of Object.entries(targetModel.weights)) {
      const val = sample[feat] ?? 0.0;
      const contrib = Math.round(val * Number(weight) * 100) / 100;
      predicted += contrib;
      breakdown[feat] = contrib;
    }

    const roundedProfit = Math.round(predicted * 100) / 100;

    return NextResponse.json(
      {
        status: 'success',
        prediction: {
          predicted_profit: roundedProfit,
          algorithm,
          k: Number(k),
          selected_features: targetModel.features,
          equation: targetModel.equation,
          breakdown,
          input_data: sample,
        },
      },
      {
        status: 200,
        headers: {
          'Access-Control-Allow-Origin': '*',
          'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
          'Access-Control-Allow-Headers': 'Content-Type',
        },
      }
    );
  } catch (error: any) {
    return NextResponse.json(
      { status: 'error', detail: error?.message || 'Failed to process prediction.' },
      { status: 500 }
    );
  }
}

export async function OPTIONS() {
  return new NextResponse(null, {
    status: 204,
    headers: {
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type',
    },
  });
}

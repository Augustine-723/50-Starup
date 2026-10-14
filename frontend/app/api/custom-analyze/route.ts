import { NextRequest, NextResponse } from 'next/server';
import { runUniversalRegressionWorkflow, parseCSV } from '../../utils/universalRegression';

export const dynamic = 'force-dynamic';

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    let {
      records,
      csv_text,
      csvText,
      targetColumn,
      target_column,
      featureColumns,
      feature_columns,
      datasetName = 'Custom Tabular Dataset',
      dataset_name,
      testSize = 0.2,
      test_size,
      randomState = 42,
      random_state,
    } = body;

    const rawCsv = csv_text || csvText;
    if ((!records || !Array.isArray(records) || records.length === 0) && rawCsv) {
      records = parseCSV(rawCsv);
    }

    if (!records || !Array.isArray(records) || records.length === 0) {
      return NextResponse.json(
        { status: 'error', message: '請提供有效的資料集紀錄 (records array 或 csv_text)' },
        { status: 400 }
      );
    }

    const finalTarget = targetColumn || target_column;
    if (!finalTarget) {
      return NextResponse.json(
        { status: 'error', message: '請指定預測目標欄位 (targetColumn)' },
        { status: 400 }
      );
    }

    let finalFeatures = featureColumns || feature_columns;
    if (!finalFeatures || !Array.isArray(finalFeatures) || finalFeatures.length === 0) {
      finalFeatures = Object.keys(records[0]).filter(k => k !== finalTarget);
    }

    const finalName = dataset_name || datasetName || 'Custom Tabular Dataset';
    const finalTestSize = Number(test_size ?? testSize) || 0.2;
    const finalRandomState = Number(random_state ?? randomState) || 42;

    const result = runUniversalRegressionWorkflow(
      records,
      finalTarget,
      finalFeatures,
      finalName,
      finalTestSize,
      finalRandomState
    );


    return NextResponse.json(result, {
      status: 200,
      headers: {
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
        'Access-Control-Allow-Headers': 'Content-Type',
      },
    });
  } catch (error: any) {
    return NextResponse.json(
      { status: 'error', message: error?.message || 'Universal Regression Analysis failed' },
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

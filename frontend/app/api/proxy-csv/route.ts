import { NextRequest, NextResponse } from 'next/server';

export const dynamic = 'force-dynamic';

// SSRF 防護：禁止存取私有網路位址與保留 IP
const BLOCKED_HOST_PATTERNS = [
  /^localhost$/i,
  /^127\./,
  /^0\./,
  /^10\./,
  /^172\.(1[6-9]|2[0-9]|3[0-1])\./,
  /^192\.168\./,
  /^169\.254\./,
  /^fc00:/i,
  /^fe80:/i,
  /^::1$/,
];

export async function GET(request: NextRequest) {
  return handleProxy(request);
}

export async function POST(request: NextRequest) {
  return handleProxy(request);
}

async function handleProxy(request: NextRequest) {
  try {
    let targetUrl = '';
    if (request.method === 'POST') {
      const body = await request.json().catch(() => ({}));
      targetUrl = body.url || '';
    } else {
      targetUrl = request.nextUrl.searchParams.get('url') || '';
    }

    if (!targetUrl || typeof targetUrl !== 'string') {
      return NextResponse.json(
        { status: 'error', message: '請提供有效的 url 參數' },
        { status: 400 }
      );
    }

    // 驗證 URL 格式
    let parsedUrl: URL;
    try {
      parsedUrl = new URL(targetUrl.trim());
    } catch {
      return NextResponse.json(
        { status: 'error', message: 'URL 格式無效，請確認是否包含 http:// 或 https://' },
        { status: 400 }
      );
    }

    // 僅允許 http 與 https
    if (parsedUrl.protocol !== 'http:' && parsedUrl.protocol !== 'https:') {
      return NextResponse.json(
        { status: 'error', message: '僅支援 HTTP 或 HTTPS 協議之公開 CSV 連結' },
        { status: 400 }
      );
    }

    // SSRF 安全檢查
    const hostname = parsedUrl.hostname.toLowerCase();
    for (const pattern of BLOCKED_HOST_PATTERNS) {
      if (pattern.test(hostname)) {
        return NextResponse.json(
          { status: 'error', message: '安全防護阻擋：禁止存取本機或私有內部網路位址 (SSRF Protected)' },
          { status: 403 }
        );
      }
    }

    // 發送請求，限制超時 10 秒
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 10000);

    const response = await fetch(parsedUrl.toString(), {
      signal: controller.signal,
      headers: {
        'User-Agent': 'Mozilla/5.0 (AIoT-HW2-Regression-Explorer/2.0)',
        'Accept': 'text/csv, text/plain, */*',
      },
      redirect: 'follow',
    });
    clearTimeout(timeoutId);

    if (!response.ok) {
      return NextResponse.json(
        {
          status: 'error',
          message: `無法取得目標 URL 資料 (HTTP ${response.status}: ${response.statusText})`,
        },
        { status: response.status }
      );
    }

    // 限制檔案大小最大 5MB
    const contentLength = response.headers.get('content-length');
    if (contentLength && parseInt(contentLength, 10) > 5 * 1024 * 1024) {
      return NextResponse.json(
        { status: 'error', message: 'CSV 檔案容量過大，超過 5MB 上限' },
        { status: 413 }
      );
    }

    const csvText = await response.text();
    if (csvText.length > 5 * 1024 * 1024) {
      return NextResponse.json(
        { status: 'error', message: 'CSV 檔案容量過大，超過 5MB 上限' },
        { status: 413 }
      );
    }

    return NextResponse.json(
      {
        status: 'success',
        url: targetUrl,
        size: csvText.length,
        csv_text: csvText,
      },
      {
        status: 200,
        headers: {
          'Access-Control-Allow-Origin': '*',
        },
      }
    );

  } catch (error: any) {
    if (error.name === 'AbortError') {
      return NextResponse.json(
        { status: 'error', message: '請求目標 URL 超時 (超過 10 秒限制)' },
        { status: 504 }
      );
    }
    return NextResponse.json(
      { status: 'error', message: error?.message || '讀取 CSV 發生未知錯誤' },
      { status: 500 }
    );
  }
}

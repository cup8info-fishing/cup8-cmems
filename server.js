// cmems-service/server.js
// Backend cap8 — espone REST API con dati reali onde Mediterraneo da Copernicus Marine.
//
// Endpoints:
//   GET /api/cmems/status              → info ultimo download (timestamp + dimensione)
//   GET /api/cmems/waves?hour=ISO      → restituisce JSON grid onde per quell'ora
//                                         (hour=now per ora corrente UTC arrotondata)
//   POST /api/cmems/refresh            → forza un nuovo download (per test/admin)
//
// Cron: ogni 6h scarica nuovi forecast (00:00, 06:00, 12:00, 18:00 UTC).
// Cache JSON: ogni "ora" estratta dal NetCDF viene cachata su disco (cache/hour_*.json),
//             servita immediatamente sui successivi GET.

const express = require('express');
const cors = require('cors');
const compression = require('compression');
const cron = require('node-cron');
const fs = require('fs');
const path = require('path');
const { spawn, execFile } = require('child_process');

// Cloud Run inietta $PORT (8080); in locale resta CMEMS_PORT o 4700.
const PORT = process.env.PORT || process.env.CMEMS_PORT || 4700;
const ROOT = __dirname;
const DATA_DIR = path.join(ROOT, 'data');
const CACHE_DIR = path.join(ROOT, 'cache');
const NC_FILE = path.join(DATA_DIR, 'forecast_med_waves.nc');
const LAST_DOWNLOAD_FILE = path.join(DATA_DIR, '.last_download');

fs.mkdirSync(DATA_DIR, { recursive: true });
fs.mkdirSync(CACHE_DIR, { recursive: true });

const PYTHON = process.platform === 'win32' ? 'python' : 'python3';

const app = express();
app.use(compression({ level: 6, threshold: 1024 }));  // gzip JSON > 1KB
app.use(cors());
app.use(express.json());

// ============================================================
// Helpers
// ============================================================
let downloadInProgress = false;

function getLastDownloadTime() {
  try {
    return fs.readFileSync(LAST_DOWNLOAD_FILE, 'utf8').trim();
  } catch {
    return null;
  }
}

function getNcFileSize() {
  try {
    return fs.statSync(NC_FILE).size;
  } catch {
    return 0;
  }
}

function nowHourIsoUtc() {
  const d = new Date();
  d.setUTCMinutes(0, 0, 0);
  return d.toISOString().slice(0, 19);  // "2026-05-25T14:00:00"
}

function cacheFileFor(hourIso) {
  // Sanifico il nome: rimuovo `:` non valido su Windows
  const safe = hourIso.replace(/[:\-]/g, '');
  return path.join(CACHE_DIR, `hour_${safe}.json`);
}

function runDownload(days = 3) {
  return new Promise((resolve, reject) => {
    if (downloadInProgress) {
      return reject(new Error('Download già in corso'));
    }
    downloadInProgress = true;
    console.log(`[cmems] avvio download (--days ${days})...`);
    const proc = spawn(PYTHON, [path.join(ROOT, 'download_cmems.py'), '--days', String(days)], {
      env: { ...process.env, PYTHONIOENCODING: 'utf-8' },
      windowsHide: true,
    });
    let stderr = '';
    proc.stdout.on('data', d => process.stdout.write(`[download] ${d}`));
    proc.stderr.on('data', d => { stderr += d; process.stderr.write(`[download:err] ${d}`); });
    proc.on('close', code => {
      downloadInProgress = false;
      if (code === 0) {
        // Invalido cache JSON (forecast aggiornato)
        try {
          for (const f of fs.readdirSync(CACHE_DIR)) {
            if (f.startsWith('hour_') && f.endsWith('.json')) fs.unlinkSync(path.join(CACHE_DIR, f));
          }
        } catch {}
        console.log('[cmems] download OK + cache JSON invalidata');
        // Re-render PNG dopo nuovo download (async, non blocca la response)
        renderPngs(72).catch(e => console.error('[render-png] post-download error:', e.message));
        resolve();
      } else {
        reject(new Error(`Python exit ${code}: ${stderr.substring(0, 500)}`));
      }
    });
  });
}

function extractHour(hourIso) {
  return new Promise((resolve, reject) => {
    const outFile = cacheFileFor(hourIso);
    if (fs.existsSync(outFile)) return resolve(outFile);
    if (!fs.existsSync(NC_FILE)) return reject(new Error('NetCDF non disponibile, eseguire download'));

    // Downsample a 0.10° (~11 km) = ~40k punti = ~3-4 MB JSON.
    // 0.05° era 14 MB (troppo per mobile). 0.10° mantiene risoluzione utile.
    const args = [path.join(ROOT, 'extract_hour.py'), hourIso, outFile, '--step', '0.10'];
    execFile(PYTHON, args, {
      env: { ...process.env, PYTHONIOENCODING: 'utf-8' },
      windowsHide: true,
      maxBuffer: 50 * 1024 * 1024,
    }, (err, stdout, stderr) => {
      if (err) return reject(new Error(`extract_hour failed: ${stderr || err.message}`));
      resolve(outFile);
    });
  });
}

// ============================================================
// Endpoints
// ============================================================
app.get('/api/cmems/status', (req, res) => {
  const lastDownload = getLastDownloadTime();
  const size = getNcFileSize();
  const cacheCount = (() => {
    try { return fs.readdirSync(CACHE_DIR).filter(f => f.startsWith('hour_')).length; }
    catch { return 0; }
  })();
  res.json({
    service: 'cmems-service',
    dataset: 'cmems_mod_med_wav_anfc_4.2km_PT1H-i',
    last_download: lastDownload,
    netcdf_size_mb: size > 0 ? +(size / (1024 * 1024)).toFixed(1) : 0,
    download_in_progress: downloadInProgress,
    cache_hours: cacheCount,
    next_cron: '00:00 / 06:00 / 12:00 / 18:00 UTC',
  });
});

app.get('/api/cmems/waves', async (req, res) => {
  try {
    let hour = req.query.hour || 'now';
    if (hour === 'now') hour = nowHourIsoUtc();
    // Validazione formato
    if (!/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}$/.test(hour)) {
      return res.status(400).json({ error: 'hour deve essere ISO UTC tipo 2026-05-25T14:00:00 oppure "now"' });
    }
    const jsonFile = await extractHour(hour);
    res.sendFile(jsonFile);
  } catch (e) {
    console.error('[waves] error:', e.message);
    res.status(503).json({ error: e.message });
  }
});

// ============================================================
// PNG endpoints (stile LaMMA) — pre-rendered server-side
// ============================================================
const META_PNG = path.join(CACHE_DIR, 'waves_meta.json');

app.get('/api/cmems/waves-png/meta', async (req, res) => {
  try {
    // Genera PNG se non esistono o NetCDF è più recente del meta
    const ncMtime = (() => { try { return fs.statSync(NC_FILE).mtimeMs; } catch { return 0; } })();
    const metaMtime = (() => { try { return fs.statSync(META_PNG).mtimeMs; } catch { return 0; } })();
    if (ncMtime > metaMtime || metaMtime === 0) {
      await renderPngs();
    }
    res.sendFile(META_PNG);
  } catch (e) {
    res.status(503).json({ error: e.message });
  }
});

app.get('/api/cmems/waves-png/:hour', (req, res) => {
  const h = parseInt(req.params.hour, 10);
  if (isNaN(h) || h < 0 || h > 71) return res.status(400).json({ error: 'hour 0-71' });
  // ?eroded=1 → versione con erosion 2 pixel per Satellite/Nautica
  const eroded = req.query.eroded === '1' || req.query.eroded === 'true';
  const suffix = eroded ? '_eroded' : '';
  const pngFile = path.join(CACHE_DIR, `waves_h${String(h).padStart(2, '0')}${suffix}.png`);
  if (!fs.existsSync(pngFile)) {
    return res.status(404).json({ error: `PNG h${h}${suffix} non disponibile, chiama /meta prima` });
  }
  res.setHeader('Cache-Control', 'public, max-age=21600');
  res.sendFile(pngFile);
});

function renderPngs(hours = 72) {
  return new Promise((resolve, reject) => {
    console.log(`[render-png] avvio rendering ${hours}h PNG...`);
    execFile(PYTHON, [path.join(ROOT, 'render_waves_png.py'), '--hours', String(hours), '--width', '3000'], {
      env: { ...process.env, PYTHONIOENCODING: 'utf-8' },
      windowsHide: true,
      maxBuffer: 50 * 1024 * 1024,
    }, (err, stdout, stderr) => {
      if (err) return reject(new Error(`render_waves_png failed: ${stderr || err.message}`));
      console.log('[render-png] ✓ done\n' + stdout.trim());
      resolve();
    });
  });
}

// GET /api/cmems/forecast?hours=24 → ritorna N ore di forecast in 1 chiamata.
// Formato compatto con array temporale per punto (= MockGridPoint frontend).
const forecastCache = new Map();  // key: hours → { mtime, data }
app.get('/api/cmems/forecast', async (req, res) => {
  try {
    const hours = Math.max(1, Math.min(72, parseInt(req.query.hours || '24', 10)));
    // step (gradi) configurabile: 72h a 0.10° pesa ~90MB → per il forecast a 72h
    // (frecce direzione + popup) il frontend chiede uno step più grosso (~0.20°).
    const step = Math.max(0.05, Math.min(1.0, parseFloat(req.query.step || '0.10')));
    const stepTag = String(step).replace('.', 'p');
    const cacheKey = `forecast_${hours}h_${stepTag}.json`;
    const cacheFile = path.join(CACHE_DIR, cacheKey);
    const ncMtime = (() => { try { return fs.statSync(NC_FILE).mtimeMs; } catch { return 0; } })();
    const cacheMtime = (() => { try { return fs.statSync(cacheFile).mtimeMs; } catch { return 0; } })();
    if (cacheMtime > ncMtime && cacheMtime > 0) {
      return res.sendFile(cacheFile);
    }
    if (!fs.existsSync(NC_FILE)) {
      return res.status(503).json({ error: 'NetCDF non disponibile, attendere primo download' });
    }
    // Estrazione fresca
    await new Promise((resolve, reject) => {
      execFile(PYTHON, [path.join(ROOT, 'extract_forecast.py'), String(hours), cacheFile, '--step', String(step)], {
        env: { ...process.env, PYTHONIOENCODING: 'utf-8' },
        windowsHide: true,
        maxBuffer: 100 * 1024 * 1024,
      }, (err, stdout, stderr) => err ? reject(new Error(stderr || err.message)) : resolve());
    });
    res.sendFile(cacheFile);
  } catch (e) {
    console.error('[forecast] error:', e.message);
    res.status(503).json({ error: e.message });
  }
});

app.post('/api/cmems/refresh', async (req, res) => {
  try {
    const days = Math.max(1, Math.min(10, parseInt(req.body?.days || '3', 10)));
    // Risposta immediata: il download è async
    res.json({ status: 'started', days });
    await runDownload(days).catch(e => console.error('[cron refresh] error:', e.message));
  } catch (e) {
    res.status(503).json({ error: e.message });
  }
});

// ============================================================
// Cron: ogni 6h alle ore 00:00, 06:00, 12:00, 18:00 UTC
// In locale (Italia CEST UTC+2) sono 02:00, 08:00, 14:00, 20:00
// ============================================================
cron.schedule('5 0,6,12,18 * * *', async () => {
  console.log('[cron] avvio download schedulato (6h tick)...');
  try { await runDownload(3); } catch (e) { console.error('[cron] download failed:', e.message); }
}, { timezone: 'UTC' });

// ============================================================
// Avvio
// ============================================================
app.listen(PORT, () => {
  console.log(`🌊 cmems-service avviato — http://localhost:${PORT}`);
  console.log(`   data:  ${DATA_DIR}`);
  console.log(`   cache: ${CACHE_DIR}`);
  console.log(`   netcdf: ${fs.existsSync(NC_FILE) ? `${(getNcFileSize() / 1024 / 1024).toFixed(1)} MB` : 'NOT YET DOWNLOADED'}`);
  console.log(`   last:  ${getLastDownloadTime() || 'mai'}`);

  // Se non c'è il NetCDF, avvio subito un download al boot
  if (!fs.existsSync(NC_FILE)) {
    console.log('[init] no NetCDF → avvio primo download...');
    runDownload(3).catch(e => console.error('[init] download failed:', e.message));
  }
});

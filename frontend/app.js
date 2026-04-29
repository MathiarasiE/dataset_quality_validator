// Minimal frontend JS to upload CSV and display validation results with charts
const fileInput = document.getElementById('fileInput')
const uploadBtn = document.getElementById('uploadBtn')
const resultCard = document.getElementById('result')
const scoreValue = document.getElementById('scoreValue')
const scoreStatus = document.getElementById('scoreStatus')
const summaryDiv = document.getElementById('summary')
const issuesDiv = document.getElementById('issues')
const reportLink = document.getElementById('reportLink')

let deductionChart = null
let severityChart = null

uploadBtn.addEventListener('click', async () => {
  if (!fileInput.files.length) {
    alert('Please select a CSV file to upload.')
    return
  }

  const file = fileInput.files[0]
  const labelCol = document.getElementById('labelCol').value || ''
  const fd = new FormData()
  fd.append('file', file)
  if (labelCol) fd.append('label_col', labelCol)

  uploadBtn.disabled = true
  uploadBtn.textContent = 'Validating...'

  try {
    const res = await fetch('/upload', { method: 'POST', body: fd })
    const data = await res.json()
    if (!res.ok) {
      let errorMsg = data.error || 'Validation failed'
      // Improve error messaging for common issues
      if (errorMsg.includes('Expected object or value')) {
        errorMsg = 'JSON file is malformed. Please check that it\'s valid JSON format.'
      } else if (errorMsg.includes('empty')) {
        errorMsg = 'File is empty. Please upload a non-empty dataset.'
      } else if (errorMsg.includes('Expecting value')) {
        errorMsg = 'Invalid JSON format. Each line in JSONL must be valid JSON.'
      }
      throw new Error(errorMsg)
    }

    // Show results
    resultCard.classList.remove('hidden')
    scoreValue.textContent = data.quality_score.final_score
    scoreStatus.textContent = data.quality_score.status
    summaryDiv.innerHTML = `<strong>Rows:</strong> ${data.rows} | <strong>Columns:</strong> ${data.columns} | <strong>Grade:</strong> ${data.quality_score.grade}`

    // Build deduction breakdown chart
    const deductions = data.quality_score.deduction_breakdown || []
    const deductionNames = deductions.map(d => d.check.replace(/_/g, ' '))
    const deductionValues = deductions.map(d => d.deduction)
    buildDeductionChart(deductionNames, deductionValues)

    // Build severity distribution chart
    const severitySummary = data.quality_score.severity_summary || {}
    buildSeverityChart(severitySummary)

    // Issues
    const checks = data.validation_checks || []
    issuesDiv.innerHTML = checks
      .filter(c => c.severity !== 'none')
      .map(c => `
        <div>
          <strong>${c.check.replace(/_/g, ' ').toUpperCase()}</strong>
          <div style="color:#97a6b8;font-size:0.85rem">${(c.suggestions||[]).slice(0,2).join(' • ')}</div>
        </div>
      `)
      .join('')

    // Report
    reportLink.href = '/report'
    reportLink.textContent = 'View full HTML report'
  } catch (err) {
    alert(err.message)
  } finally {
    uploadBtn.disabled = false
    uploadBtn.textContent = 'Run Validation'
  }
})

function buildDeductionChart(labels, values) {
  const ctx = document.getElementById('deductionChart')
  if (deductionChart) deductionChart.destroy()
  deductionChart = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: labels,
      datasets: [{
        label: 'Points Deducted',
        data: values,
        backgroundColor: 'rgba(248, 113, 113, 0.7)',
        borderColor: '#f87171',
        borderWidth: 1
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: true,
      plugins: {
        legend: { display: false }
      },
      scales: {
        y: {
          beginAtZero: true,
          ticks: { color: '#94a3b8' },
          grid: { color: 'rgba(255,255,255,0.05)' }
        },
        x: {
          ticks: { color: '#94a3b8' },
          grid: { display: false }
        }
      }
    }
  })
}

function buildSeverityChart(severitySummary) {
  const ctx = document.getElementById('severityChart')
  const severities = ['high', 'medium', 'low', 'none']
  const colors = { high: '#ef4444', medium: '#f59e0b', low: '#3b82f6', none: '#22c55e' }
  const severityValues = severities.map(s => severitySummary[s] || 0)
  const severityColors = severities.map(s => colors[s])

  if (severityChart) severityChart.destroy()
  severityChart = new Chart(ctx, {
    type: 'doughnut',
    data: {
      labels: severities.map(s => s.charAt(0).toUpperCase() + s.slice(1)),
      datasets: [{
        data: severityValues,
        backgroundColor: severityColors,
        borderColor: '#0f1724',
        borderWidth: 2
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: true,
      plugins: {
        legend: {
          position: 'bottom',
          labels: { color: '#94a3b8', padding: 12 }
        }
      }
    }
  })
}

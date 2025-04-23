// Global variables
let eventSource = null;
let currentFile = null;
let queueId = null;
let currentFileName = null;

// Initialize on document load
document.addEventListener('DOMContentLoaded', function() {
    // File input change handler
    document.getElementById('audioFile').addEventListener('change', function(e) {
        const file = e.target.files[0];
        if (file) {
            document.getElementById('fileName').textContent = file.name;
            currentFile = file;
            
            // Enable the analyze button
            document.getElementById('analyzeBtn').disabled = false;
            
            // Set the audio source for preview
            const audioPlayer = document.getElementById('audioPlayer');
            audioPlayer.src = URL.createObjectURL(file);
        }
    });
    
    // Disable analyze button initially
    document.getElementById('analyzeBtn').disabled = true;
});

// Function to start the analysis
function analyze() {
    if (!currentFile) {
        alert('Lütfen bir müzik dosyası seçiniz.');
        return;
    }
    
    // Show progress section
    document.getElementById('progressSection').classList.remove('hidden');
    document.getElementById('results').classList.add('hidden');
    
    // Reset progress bar
    const progressBar = document.getElementById('progress');
    progressBar.style.width = '0%';
    progressBar.textContent = '0%';
    
    // Create form data
    const formData = new FormData();
    formData.append('file', currentFile);
    
    // Submit the file for analysis
    fetch('/analyze', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            showError(data.error);
            return;
        }
        
        // Store the queue ID for progress tracking
        queueId = data.queue_id;
        
        // Start listening for progress updates
        startProgressTracking(queueId);
        
        // Store the filename for retrieving results later
        currentFileName = data.filename;
    })
    .catch(error => {
        showError('Analysis request failed: ' + error);
    });
}

// Function to track progress updates
function startProgressTracking(queueId) {
    // Close any existing event source
    if (eventSource) {
        eventSource.close();
    }
    
    // Create a new event source for SSE
    eventSource = new EventSource(`/progress?id=${queueId}`);
    
    // Handle incoming progress messages
    eventSource.onmessage = function(event) {
        const progress = parseInt(event.data);
        updateProgressBar(progress);
        
        // If progress is 100%, get the results
        if (progress === 100) {
            // Close the event source
            eventSource.close();
            eventSource = null;
            
            // Fetch the results after a short delay
            setTimeout(() => fetchResults(currentFileName), 500);
        }
    };
    
    // Handle errors
    eventSource.onerror = function() {
        eventSource.close();
        eventSource = null;
        showError('Lost connection to the server');
    };
}

// Function to update the progress bar
function updateProgressBar(progress) {
    const progressBar = document.getElementById('progress');
    progressBar.style.width = `${progress}%`;
    progressBar.textContent = `${progress}%`;
}

// Function to fetch and display analysis results
function fetchResults(filename) {
    fetch(`/result/${filename}`)
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                showError(data.error);
                return;
            }
            
            // Hide progress section and show results
            document.getElementById('progressSection').classList.add('hidden');
            document.getElementById('results').classList.remove('hidden');
            
            // Display basic results
            displayBasicResults(data);
            
            // Display timbre results
            displayTimbreResults(data);
            
            // Create visualizations
            createVisualizations(data);
            
            // Format and display duration
            displayDuration(data.duration);
        })
        .catch(error => {
            showError('Failed to fetch results: ' + error);
        });
}

// Function to display basic analysis results
function displayBasicResults(data) {
    // Determine system and tonality
    const isWestern = data.tonality.is_western;
    const systemText = isWestern ? 'Batı Müziği' : 'Doğu Müziği';
    const tonalityText = isWestern ? data.tonality.western_tonality : data.tonality.eastern_makam;
    const altTonalityText = isWestern ? 
        `Alternatif olarak: ${data.tonality.eastern_makam} (Doğu)` : 
        `Alternatif olarak: ${data.tonality.western_tonality} (Batı)`;
    
    // Display system and tonality
    document.getElementById('system').textContent = systemText;
    document.getElementById('tonality').textContent = tonalityText;
    document.getElementById('tonalityAlt').textContent = altTonalityText;
    
    // Set confidence meter
    const confidence = isWestern ? 
        data.tonality.western_confidence * 100 : 
        data.tonality.eastern_confidence * 100;
    document.getElementById('systemConfidence').style.width = `${confidence}%`;
    document.getElementById('confidenceValue').textContent = `${Math.round(confidence)}%`;
    
    // Display tempo with description
    const tempo = Math.round(data.tempo);
    document.getElementById('tempo').textContent = `${tempo} BPM`;
    
    // Add tempo description
    let tempoDescription = '';
    if (tempo < 60) {
        tempoDescription = 'Çok Yavaş (Largo)';
    } else if (tempo < 76) {
        tempoDescription = 'Yavaş (Adagio)';
    } else if (tempo < 108) {
        tempoDescription = 'Orta (Andante)';
    } else if (tempo < 120) {
        tempoDescription = 'Hızlı (Moderato)';
    } else if (tempo < 168) {
        tempoDescription = 'Çok Hızlı (Allegro)';
    } else {
        tempoDescription = 'Aşırı Hızlı (Presto)';
    }
    document.getElementById('tempoDescription').textContent = tempoDescription;
    
    // Display rhythm pattern
    document.getElementById('rhythmPattern').textContent = data.rhythm_pattern;
    
    // Display beat regularity
    const regularity = Math.round(data.beat_regularity * 100);
    let regularityDesc = '';
    if (regularity > 80) {
        regularityDesc = 'Çok düzenli vuruş';
    } else if (regularity > 60) {
        regularityDesc = 'Düzenli vuruş';
    } else if (regularity > 40) {
        regularityDesc = 'Orta düzeyde düzenli';
    } else {
        regularityDesc = 'Düzensiz vuruş';
    }
    document.getElementById('beatRegularity').textContent = `${regularityDesc} (${regularity}%)`;
}

// Function to display timbre analysis results
function displayTimbreResults(data) {
    if (!data.timbre) return;
    
    // Display instrument family
    document.getElementById('instrumentFamily').textContent = data.timbre.instrument_family;
    
    // Display brightness
    const brightness = Math.round(data.timbre.brightness * 100);
    document.getElementById('brightnessMeter').style.width = `${brightness}%`;
    document.getElementById('brightnessValue').textContent = `${brightness}%`;
    
    // Display richness (normalize to 0-100%)
    const richness = Math.min(100, Math.round((data.timbre.richness / 10) * 100));
    document.getElementById('richnessMeter').style.width = `${richness}%`;
    document.getElementById('richnessValue').textContent = `${richness}%`;
}

// Function to format and display duration
function displayDuration(seconds) {
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = Math.round(seconds % 60);
    const formattedDuration = `${minutes}:${remainingSeconds.toString().padStart(2, '0')}`;
    document.getElementById('duration').textContent = formattedDuration;
}

// Function to create data visualizations
function createVisualizations(data) {
    // Create frequency distribution graph
    createFrequencyGraph(data.frequencies);
    
    // Create tempo comparison graph
    createTempoGraph(data.tempo);
    
    // Create MFCC graph if available
    if (data.timbre && data.timbre.mfcc_features) {
        createMFCCGraph(data.timbre.mfcc_features);
    }
}

// Function to create frequency distribution graph
function createFrequencyGraph(frequencies) {
    // Filter out zero frequencies and limit to meaningful ones (e.g., first 50)
    const nonZeroFreqs = frequencies.filter(f => f > 0).slice(0, 50);
    
    // Prepare data for visualization
    const trace = {
        y: nonZeroFreqs,
        type: 'scatter',
        mode: 'lines+markers',
        marker: {
            color: '#3498db',
            size: 8
        },
        line: {
            color: '#2980b9',
            width: 2
        },
        name: 'Frekans Dağılımı'
    };
    
    // Layout configuration
    const layout = {
        title: 'Parça Frekans Dağılımı',
        xaxis: {
            title: 'Zaman'
        },
        yaxis: {
            title: 'Frekans (Hz)'
        },
        margin: { t: 50, r: 20, l: 60, b: 50 },
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        font: {
            family: 'Poppins, sans-serif'
        }
    };
    
    // Create the plot
    Plotly.newPlot('frequencyGraph', [trace], layout, {responsive: true});
}

// Function to create tempo visualization
function createTempoGraph(tempo) {
    // Define common tempo benchmarks
    const tempos = [
        { name: 'Çok Yavaş (Largo)', value: 50 },
        { name: 'Yavaş (Adagio)', value: 70 },
        { name: 'Orta (Andante)', value: 90 },
        { name: 'Hızlı (Moderato)', value: 112 },
        { name: 'Çok Hızlı (Allegro)', value: 140 },
        { name: 'Aşırı Hızlı (Presto)', value: 180 }
    ];
    
    // Find where this tempo falls
    let closestIndex = 0;
    let minDiff = Math.abs(tempo - tempos[0].value);
    
    for (let i = 1; i < tempos.length; i++) {
        const diff = Math.abs(tempo - tempos[i].value);
        if (diff < minDiff) {
            minDiff = diff;
            closestIndex = i;
        }
    }
    
    // Prepare data for the bar chart
    const tempoValues = tempos.map(t => t.value);
    const tempoNames = tempos.map(t => t.name);
    
    // Add the current tempo
    tempoValues.push(tempo);
    tempoNames.push('Bu Parça');
    
    // Colors array (highlighting the current tempo)
    const colors = tempos.map(() => '#3498db');
    colors.push('#e74c3c'); // Highlight the current tempo
    
    // Create the bar chart
    const trace = {
        x: tempoNames,
        y: tempoValues,
        type: 'bar',
        marker: {
            color: colors
        }
    };
    
    // Layout configuration
    const layout = {
        title: 'Tempo Karşılaştırma',
        xaxis: {
            title: '',
            tickangle: -45
        },
        yaxis: {
            title: 'BPM (Vuruş/Dakika)'
        },
        margin: { t: 50, r: 20, l: 60, b: 120 },
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        font: {
            family: 'Poppins, sans-serif'
        }
    };
    
    // Create the plot
    Plotly.newPlot('tempoGraph', [trace], layout, {responsive: true});
}

// Function to create MFCC visualization
function createMFCCGraph(mfccFeatures) {
    // Prepare data for visualization
    const trace = {
        y: mfccFeatures,
        type: 'bar',
        marker: {
            color: mfccFeatures.map((v, i) => {
                // Generate color gradient based on value
                const h = 240 * (1 - Math.min(1, Math.max(0, (v + 20) / 40)));
                return `hsl(${h}, 70%, 50%)`;
            })
        }
    };
    
    // Layout configuration
    const layout = {
        title: 'MFCC Özellikleri (Tını Analizi)',
        xaxis: {
            title: 'MFCC Katsayıları'
        },
        yaxis: {
            title: 'Değer'
        },
        margin: { t: 50, r: 20, l: 60, b: 50 },
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        font: {
            family: 'Poppins, sans-serif'
        }
    };
    
    // Create the plot
    Plotly.newPlot('mfccGraph', [trace], layout, {responsive: true});
}

// Function to display errors
function showError(message) {
    alert('Hata: ' + message);
    document.getElementById('progressSection').classList.add('hidden');
}
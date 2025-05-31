<script>
    import { onMount } from 'svelte';
    import { apiFetch } from '$lib/api.js';

    let futurePredictedMatches = [];
    let isLoading = true;
    let error = null;
    let predictionInProgressFor = {};

    const limit = 25;
    let currentPage = 0;
    const legendItems = [
        { abbr: 'H/D/A', full: 'Prawdopodobieństwo: Wygrana Gospodarzy / Remis / Wygrana Gości (w %)' },
        { abbr: 'xG H:A', full: 'Oczekiwane Gole (Expected Goals): Gospodarze : Goście' },
        { abbr: 'LR', full: 'Regresja Logistyczna (Klasyfikator) / Liniowa (Regresor)' },
        { abbr: 'RFC', full: 'Random Forest Classifier (Klasyfikator)' },
        { abbr: 'RFR', full: 'Random Forest Regressor (Regresor)' },
        { abbr: 'XGB', full: 'XGBoost Classifier (Klasyfikator)' },
        { abbr: 'XGBR', full: 'XGBoost Regressor (Regresor)' },
        { abbr: 'SVC', full: 'Support Vector Classifier (Klasyfikator)' },
        { abbr: 'SVR', full: 'Support Vector Regressor (Regresor)' },
    ];

    async function fetchFuturePredictions() {
        isLoading = true; error = null;
        const skip = currentPage * limit;
        try {
            const endpoint = `/predicted_matches?skip=${skip}&limit=${limit}&sort_by_target_date=asc&only_future=true`;
            const data = await apiFetch(endpoint);
            futurePredictedMatches = data && Array.isArray(data) ? data : [];
            console.log("Fetched future predictions:", futurePredictedMatches);
        } catch (err) {
            console.error("Error fetching future predicted matches:", err);
            error = "Nie udało się załadować listy przyszłych predykcji.";
            futurePredictedMatches = [];
        } finally {
            isLoading = false;
        }
    }

    onMount(fetchFuturePredictions);

    function navigatePage(direction) {
        let newPage = currentPage;
        if (direction === 1 && futurePredictedMatches.length === limit) {
            newPage++;
        } else if (direction === -1 && currentPage > 0) {
            newPage--;
        }
        if (newPage !== currentPage) {
            currentPage = newPage;
            fetchFuturePredictions();
        }
    }

    async function handleDeletePrediction(predictionId) {
        if(!confirm(`Czy na pewno chcesz usunąć mecz do predykcji o ID: ${predictionId}`)) {
            return;
        }
        try {
            await apiFetch(`/predicted_matches/${predictionId}`, {method: 'DELETE'});
            futurePredictedMatches = futurePredictedMatches.filter(p => p.id !== predictionId);
            alert('Mecz do predykcji został usunięty');
        } catch(err) {
            console.error("Error deleting prediction: ", err);
            alert(`Błąd usuwania predykcji: ${err.message}`);
            error = `Nie udało się usunąć predykcji: ${err.message}`;
        }
    }

    function formatProbabilities(pMatch, modelPrefix) {
        const home = pMatch[`home_win_probability_${modelPrefix}`];
        const draw = pMatch[`draw_probability_${modelPrefix}`];
        const away = pMatch[`away_win_probability_${modelPrefix}`];
        if (home === null || home === undefined) return 'N/A';
        return `${home.toFixed(1)}% | ${draw.toFixed(1)}% | ${away.toFixed(1)}%`;
    }
    function formatXG(pMatch, modelPrefix) {
        const homeXG = pMatch[`home_expected_goals_${modelPrefix}`];
        const awayXG = pMatch[`away_expected_goals_${modelPrefix}`];
        if (homeXG === null || homeXG === undefined) return 'N/A';
        return `${homeXG.toFixed(2)} : ${awayXG.toFixed(2)}`;
    }

    async function handlePredictFixture(matchId) {
        if(!matchId) return;
        predictionInProgressFor[matchId] = true;
        let errorForThisMatch = null;

        try{
            const predictionResult = await apiFetch(`/predicted_matches/predict/${matchId}`, {
               method: 'POST'
            })
            console.log(`Predykcja dla meczu ID ${matchId} zakończona:`, predictionResult)
            alert(`Predykcja dla meczu ID ${matchId} została wygenerowana!`);
            fetchFuturePredictions();
        } catch(err) {
            console.error(`Błąd podczas przewidywania meczu ID ${matchId}`, err);
            errorForThisMatch = err.message;
            alert(`Błąd dla przewidywania dla meczu ID ${matchId}: ${err.message}`);
        } finally {
            predictionInProgressFor[matchId] = false;
        }
    }



</script>

<svelte:head>
    <title>Przyszłe Predykcje - AI Predictor</title>
</svelte:head>

<section class="page-content">
    <h1>Wygenerowane Predykcje (Mecze Przyszłe / Zaplanowane)</h1>

    <div class="content-grid">
        <div class="predictions-table-container">
            {#if isLoading}
                <p>Ładowanie predykcji...</p>
            {:else if error}
                <p class="error-message">{error}</p>
            {:else if futurePredictedMatches.length > 0}
                <div class="table-wrapper">
                    <table>
                        <thead>
                            <tr>
                                <th class="col-teams">Mecz (Gospodarz vs Gość)</th>
                                <th class="col-prediction">Predykcja LR (H/D/A | xG)</th>
                                <th class="col-prediction">Predykcja RF (H/D/A | xG)</th>
                                <th class="col-prediction">Predykcja XGB (H/D/A | xG)</th>
                                <th class="col-prediction">Predykcja SV (H/D/A | xG)</th>
                                <th class="col-actions">Akcje</th>
                            </tr>
                        </thead>
                        <tbody>
                            {#each futurePredictedMatches as pMatch (pMatch.id)}
                                <tr class="prediction-row">
                                    <td class="col-teams">
                                        <span class="team home">{pMatch.home_team?.team_name || 'Gospodarz?'}</span>
                                        <span class="vs">vs</span>
                                        <span class="team away">{pMatch.away_team?.team_name || 'Gość?'}</span>
                                    </td>
                                    <td class="col-prediction">
                                        <div class="prob">{formatProbabilities(pMatch, 'lr')}</div>
                                        <div class="xg">{formatXG(pMatch, 'lr')}</div>
                                    </td>
                                    <td class="col-prediction">
                                        <div class="prob">{formatProbabilities(pMatch, 'rfc')}</div>
                                        <div class="xg">{formatXG(pMatch, 'rfr')}</div>
                                    </td>
                                    <td class="col-prediction">
                                        <div class="prob">{formatProbabilities(pMatch, 'xgb')}</div>
                                        <div class="xg">{formatXG(pMatch, 'xgb')}</div>
                                    </td>
                                    <td class="col-prediction">
                                        <div class="prob">{formatProbabilities(pMatch, 'svc')}</div>
                                        <div class="xg">{formatXG(pMatch, 'svr')}</div>
                                    </td>
                                    <td class="col-actions">
                                        {#if !pMatch.is_predicted}
                                            <button 
                                                class="predict-button-on-list"
                                                on:click={() => handlePredictFixture(pMatch.id)}
                                                disabled={predictionInProgressFor[pMatch.id]}
                                            >
                                                {#if predictionInProgressFor[pMatch.id]}
                                                    Przetwarzanie...
                                                {:else}
                                                    Predykcja
                                                {/if}
                                            </button>
                                        {/if}
                                        <button class="action-button delete-button" on:click={() => handleDeletePrediction(pMatch.id)}>
                                            Usuń
                                        </button>
                                    </td>
                                </tr>
                            {/each}
                        </tbody>
                    </table>
                </div>
                <div class="pagination-controls">
                    <button on:click={() => navigatePage(-1)} disabled={currentPage === 0 || isLoading}>&laquo; Poprzednia</button>
                    <span>Strona {currentPage + 1}</span>
                    <button on:click={() => navigatePage(1)} disabled={futurePredictedMatches.length < limit || isLoading}>Następna &raquo;</button>
                </div>
            {:else}
                <p>Brak wygenerowanych predykcji dla przyszłych meczów do wyświetlenia.</p>
            {/if}
        </div>

        <aside class="sidebar-right card">
            <h2>Legenda</h2>
            <ul>
                {#each legendItems as item (item.abbr)}
                    <li><strong>{item.abbr}</strong>: {item.full}</li>
                {/each}
            </ul>
        </aside>
    </div>
</section>

<style>
    .page-content h1 {
        color: #333;
        border-bottom: 2px solid #1abc9c;
        padding-bottom: 0.5rem;
        margin-bottom: 1.5rem;
    }
    .content-grid {
        display: grid;
        grid-template-columns: minmax(0, 3fr) minmax(0, 1fr);
        gap: 2rem;
    }
    .predictions-table-container h2 { /* ... */ }
    .table-wrapper { overflow-x: auto; }
    table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.85rem;
        white-space: nowrap;
    }
    th, td {
        text-align: left;
        padding: 0.6rem 0.4rem;
        border-bottom: 1px solid #ecf0f1;
        vertical-align: middle;
    }
    th {
        background-color: #f2f4f6;
        font-weight: 600;
        color: #555;
        position: sticky;
        top: 0;
        z-index: 1;
    }
    tbody tr:hover { background-color: #f9f9f9; }

    .col-date { min-width: 100px; text-align: center; }
    .col-teams { min-width: 220px;  font-weight: bold;}
    .col-teams .team { display: block;}
    .col-teams .team.home { font-weight: 500;  font-weight: bold;}
    .col-teams .vs { display: block; text-align: center; font-size: 0.8em; color: #777; margin: 0.1em 0; }
    
    .col-prediction { 
        min-width: 160px;
        text-align: center;
    }
    .col-prediction .prob { font-size: 0.9em; }
    .col-prediction .xg { font-size: 0.8em; color: #555; margin-top: 0.2em; }

    .col-actions { min-width: 120px; text-align: center; }
    .action-button {
        padding: 0.3rem 0.6rem;
        font-size: 0.8rem;
        background-color: #3498db;
        color: white;
        border-radius: 3px;
        text-decoration: none;
        white-space: nowrap;
    }
    .action-button:hover { background-color: #2980b9; }

    .sidebar-right { align-self: flex-start; }
    .legend-sidebar h2 {}
    .legend-sidebar ul { list-style: none; padding: 0; font-size: 0.9rem;}
    .legend-sidebar li { margin-bottom: 0.5rem; }
    .legend-sidebar strong { color: #2c3e50; }

    .pagination-controls {}
    .error-message {}
    .card {
        background: white;
        padding: 1.5rem;
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }

    @media (max-width: 1200px) {
        .content-grid {
            grid-template-columns: 1fr;
        }
        .sidebar-right {
            margin-top: 2rem;
        }
    }

    .delete-button {
        background-color: #e74c3c;
    }
    .delete-button:hover {
        background-color: #c0392b;
    }

    .predict-button-on-list {
        padding: 0.4rem 0.8rem;
        font-size: 0.9em;
    }
</style>
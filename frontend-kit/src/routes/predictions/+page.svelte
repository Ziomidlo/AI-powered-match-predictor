<script>
    import { onMount } from 'svelte';
    import { apiFetch } from '$lib/api.js'; // lub '../lib/api.js'

    let futurePredictedMatches = [];
    let isLoading = true;
    let error = null;

    const limit = 50;
    let currentPage = 0;

    // Czy ta strona powinna mieć selektor sezonu?
    // To zależy, czy Twój endpoint /api/predicted_matches wspiera filtrowanie po season_id.
    // Jeśli predykcje nie są sztywno powiązane z sezonem (np. są to też predykcje
    // dla meczów hipotetycznych), to selektor sezonu może tu nie pasować.
    // Na razie zakładam, że nie ma tu selektora sezonu, tylko paginowana lista wszystkich predykcji.
    // Jeśli chcesz filtrować po sezonie, musisz dodać logikę podobną do PlayedMatchesPage.

    async function fetchFuturePredictions() {
        isLoading = true; error = null;
        const skip = currentPage * limit;
        try {
            // Zakładamy, że /predicted_matches zwraca listę predykcji, które mogą być "przyszłe"
            // i wspiera paginację oraz sortowanie (np. po dacie meczu, którego dotyczy predykcja, lub dacie utworzenia predykcji)
            // Jeśli PredictedMatch ma pole np. 'target_match_date' i 'is_played_flag_from_source_match'
            const endpoint = `/predicted_matches?skip=${skip}&limit=${limit}&sort_by_target_date=asc&only_future=true`; // PRZYKŁADOWE parametry
            const data = await apiFetch(endpoint);
            futurePredictedMatches = data || [];
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
        if (direction === 1 && futurePredictedMatches.length === limit) currentPage++;
        else if (direction === -1 && currentPage > 0) currentPage--;
        fetchFuturePredictions();
    }
</script>

<svelte:head>
    <title>Przyszłe Predykcje - AI Predictor</title>
</svelte:head>

<section class="future-predictions-content">
    <h1>Wygenerowane Predykcje (Mecze Przyszłe)</h1>

    {#if isLoading}
        <p>Ładowanie predykcji...</p>
    {:else if error}
        <p class="error-message">{error}</p>
    {:else if futurePredictedMatches.length > 0}
        <ul class="match-list prediction-list">
            {#each futurePredictedMatches as pMatch (pMatch.id)}
                <li class="match-item">
                    <span class="team home-team">{pMatch.home_team?.team_name || 'Gospodarz?'}</span>
                    <span class="prediction-probabilities" title="P(Wygrana Gospodarza) | P(Remis) | P(Wygrana Gościa) - Model LR">
                        LR: {pMatch.home_win_probability_lr?.toFixed(1)}% | 
                        {pMatch.draw_probability_lr?.toFixed(1)}% | 
                        {pMatch.away_win_probability_lr?.toFixed(1)}%
                    </span>
                    <span class="team away-team">{pMatch.away_team?.team_name || 'Gość?'}</span>
                    <span class="prediction-xg" title="xG Gospodarz - xG Gość (Model LR)">
                        xG: {pMatch.home_expected_goals_lr?.toFixed(2)} - {pMatch.away_expected_goals_lr?.toFixed(2)}
                    </span>
                    {#if pMatch.target_match_date}
                         <span class="match-date">{new Date(pMatch.target_match_date).toLocaleDateString('pl-PL')}</span>
                    {/if}
                    <a href="/predict?home_team_id={pMatch.home_team_id}&away_team_id={pMatch.away_team_id}&source_predicted_match_id={pMatch.id}" 
                       class="predict-link-button-small">
                        Szczegóły/Nowa
                    </a>
                </li>
            {/each}
        </ul>
        <div class="pagination-controls">
            <button on:click={() => navigatePage(-1)} disabled={currentPage === 0 || isLoading}>&laquo; Poprzednia</button>
            <span>Strona {currentPage + 1}</span>
            <button on:click={() => navigatePage(1)} disabled={futurePredictedMatches.length < limit || isLoading}>Następna &raquo;</button>
        </div>
    {:else}
        <p>Brak wygenerowanych predykcji dla przyszłych meczów.</p>
    {/if}
</section>

<style>
    /* ... style dla .future-predictions-content ... */
    /* ... .match-list, .match-item, .list-header (jeśli dodasz), .pagination-controls ... */
    /* Style z poprzedniej odpowiedzi dla .match-item, .team, .predict-link-button-small mogą być przydatne */
    .match-item {
        /* ... */
        /* Dostosuj flex-basis dla nowych kolumn */
    }
    .prediction-probabilities, .prediction-xg {
        /* ... style ... */
        text-align: center;
        font-size: 0.9em;
        min-width: 150px; /* Aby się zmieściły procenty */
    }
</style>
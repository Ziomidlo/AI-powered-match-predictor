<script>
    import { onMount } from 'svelte';
    import { apiFetch } from '$lib/api.js'; 
    import SeasonSelector from '../../components/SeasonSelector.svelte'

    let allSeasons = [];
    let selectedSeasonId = null;
    let playedMatches = [];
    
    let isLoadingSeasons = true;
    let isLoadingMatches = false;
    let error = null;

    const limit = 50;
    let currentPage = 0;

    async function fetchSeasons() {
        isLoadingSeasons = true; error = null;
        try {
            const data = await apiFetch('/seasons?sort_by_name=desc');
            if (data && data.length > 0) {
                allSeasons = data.map(s => ({ id: s.id, name: s.season }));
                if (allSeasons.length > 0) {
                    selectedSeasonId = allSeasons[0].id;
                }
            }
        } catch (err) {
            console.error("Error fetching seasons:", err);
            error = "Nie udało się załadować listy sezonów.";
        } finally {
            isLoadingSeasons = false;
        }
    }

    onMount(fetchSeasons);

    async function fetchPlayedMatchesData() {
        if (selectedSeasonId === null || selectedSeasonId === undefined) {
            playedMatches = [];
            return;
        }
        isLoadingMatches = true; error = null;
        const skip = currentPage * limit;
        try {
            // Zakładamy, że /api/matches domyślnie zwraca rozegrane lub ma parametr status=played
            const endpoint = `/matches?season_id=${selectedSeasonId}&skip=${skip}&limit=${limit}`;
            const data = await apiFetch(endpoint);
            playedMatches = data || [];
        } catch (err) {
            console.error("Error fetching played matches:", err);
            error = "Nie udało się załadować rozegranych meczów.";
            playedMatches = [];
        } finally {
            isLoadingMatches = false;
        }
    }

    $: if (selectedSeasonId !== null && allSeasons.length > 0) {
        currentPage = 0; // Resetuj stronę przy zmianie sezonu
        console.log("WYbrany sezon:", selectedSeasonId)
        fetchPlayedMatchesData();
    }

    function navigatePage(direction) {
        if (direction === 1 && playedMatches.length === limit) currentPage++;
        else if (direction === -1 && currentPage > 0) currentPage--;
        fetchPlayedMatchesData();
    }

    
</script>

<svelte:head>
    <title>Rozegrane Mecze - AI Predictor</title>
</svelte:head>

<section class="matches-page-content">
    <h1>Rozegrane Mecze</h1>

    {#if isLoadingSeasons}
        <p>Ładowanie sezonów...</p>
    {:else if allSeasons.length > 0}
        <SeasonSelector 
            availableSeasons={allSeasons} 
            bind:selectedSeasonId={selectedSeasonId}
        />
    {:else if error}
        <p class="error-message">{error}</p>
    {/if}

    {#if selectedSeasonId !== null}
        {#if isLoadingMatches}
            <p>Ładowanie meczów...</p>
        {:else if error && playedMatches.length === 0}
            <p class="error-message">{error}</p>
        {:else if playedMatches.length > 0}
            <ul class="match-list">
                <li class="list-header"><span>Data</span><span>Gospodarz</span><span>Wynik</span><span>Gość</span><span>Akcje</span></li>
                {#each playedMatches as match (match.id)}
                    <li class="match-item">
                        <span class="match-date">{new Date(match.match_date).toLocaleDateString('pl-PL', { day: '2-digit', month: '2-digit', year: 'numeric' })}</span>
                        <span class="team home-team">{match.home_team.team_name}</span>
                        <strong class="score">{match.home_xG} : {match.away_xG}</strong>
                        <strong class="score">{match.home_goals} - {match.away_goals}</strong>
                        <span class="team away-team">{match.away_team.team_name}</span>
                    </li>
                {/each}
            </ul>
            <div class="pagination-controls">
                <button on:click={() => navigatePage(-1)} disabled={currentPage === 0 || isLoadingMatches}>&laquo; Poprzednia</button>
                <span>Strona {currentPage + 1}</span>
                <button on:click={() => navigatePage(1)} disabled={playedMatches.length < limit || isLoadingMatches}>Następna &raquo;</button>
            </div>
        {:else}
            <p>Brak rozegranych meczów dla wybranego sezonu.</p>
        {/if}
    {/if}
</section>

<style>
    /* ... (style z poprzedniej odpowiedzi dla .matches-page, .view-selector, .match-list, .match-item, itd.) ... */
    /* Dodatkowe lub zmodyfikowane style */
    .controls-wrapper {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 1.5rem;
        flex-wrap: wrap; /* Na mniejszych ekranach */
    }
    .list-header {
        display: flex;
        align-items: center;
        font-weight: bold;
        padding: 0.8rem 1rem;
        background-color: #f8f9fa;
        border-bottom: 2px solid #e0e0e0;
        margin-bottom: 0.5rem;
    }
    .list-header span { flex: 1; text-align: center; }
    .list-header span:nth-child(1) { flex-basis: 160px; text-align: left;} /* Data */
    .list-header span:nth-child(2) { flex-basis: 30%; text-align: right; padding-right: 0.5rem;} /* Home Team */
    .list-header span:nth-child(3) { flex-basis: 15%; font-size: 1.1em;} /* Score / Prediction % */
    .list-header span:nth-child(4) { flex-basis: 30%; text-align: left; padding-left: 0.5rem;} /* Away Team */
    .list-header span:nth-child(5) { flex-basis: 150px; text-align: right;} /* Akcje lub xG */
    
    .match-item > * {
        padding: 0 0.25rem; /* Mniejszy padding dla elementów w wierszu */
    }
    .match-item .match-date { flex-basis: 160px; text-align: left; }
    .match-item .team.home-team { flex-basis: 30%; text-align: right; }
    .match-item .score, .match-item .vs-separator, .match-item .prediction-probabilities, .match-item .prediction-xg {
        flex-basis: 15%; text-align: center;
    }
    .match-item .team.away-team { flex-basis: 30%; text-align: left;}
    .match-item .predict-link-button-small, .match-item .badge-predicted {
        flex-basis: 150px; 
        text-align: right;
        margin-left: auto; /* Dopasowanie, jeśli było wcześniej */
    }
    .predict-link-button-small { /* Mniejsza wersja przycisku */
        padding: 0.3rem 0.6rem;
        font-size: 0.8rem;
        background-color: #5cb85c;
        color: white;
        border-radius: 3px;
        text-decoration: none;
        white-space: nowrap;
    }
    .predict-link-button-small:hover{
        background-color: #4cae4c;
    }
    .prediction-list .prediction-probabilities { font-size: 0.9em; color: #337ab7; }
    .prediction-list .prediction-xg { font-size: 0.9em; color: #5cb85c; }

</style>
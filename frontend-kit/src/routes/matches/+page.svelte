<script>
    import { onMount, afterUpdate } from 'svelte';
    import { apiFetch } from '$lib/api.js'; 
    import SeasonSelector from '../../components/SeasonSelector.svelte'

    let allSeasons = [];
    let selectedSeasonId = null;
    let playedMatches = [];
    let lastSeasonId = null;
    
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
            const endpoint = `/matches/season/${selectedSeasonId}?skip=${skip}&limit=${limit}`;
            const data = await apiFetch(endpoint);
            console.log("Endpoint: ", endpoint)
            playedMatches = data || [];
        } catch (err) {
            console.error("Error fetching played matches:", err);
            error = "Nie udało się załadować rozegranych meczów.";
            playedMatches = [];
        } finally {
            isLoadingMatches = false;
        }
    }

    afterUpdate(() => {
    if (selectedSeasonId && selectedSeasonId !== lastSeasonId) {
        lastSeasonId = selectedSeasonId;
        console.log("Wybrany sezon: " ,selectedSeasonId )
        currentPage = 0;
        fetchPlayedMatchesData();
    }
    })


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
                <li class="list-header match-item-grid">
                    <span class="header-date">Data</span>
                    <span class="header-home-team">Gospodarz</span>
                    <span class="header-score">Wynik</span>
                    <span class="header-away-team">Gość</span>
                    <span class="header-xg">xG (H:A)</span>
                </li>
                {#each playedMatches as match (match.id)}
                    <li class="match-item match-item-grid">
                        <span class="match-date">{new Date(match.match_date).toLocaleDateString('pl-PL', { day: '2-digit', month: '2-digit', year: 'numeric' })}</span>
                        <span class="team home-team">{match.home_team.team_name}</span>
                        <strong class="score actual-score">{match.home_goals} - {match.away_goals}</strong>
                        <span class="team away-team">{match.away_team.team_name}</span>
                        <span class="score xg-score">
                            {match.home_xG !== null && match.home_xG !== undefined ? match.home_xG.toFixed(2) : 'N/A'} : 
                            {match.away_xG !== null && match.away_xG !== undefined ? match.away_xG.toFixed(2) : 'N/A'}
                        </span>
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

    .match-list {
        list-style: none;
        padding: 0;
    }

    .match-item-grid {
        display: grid;
        grid-template-columns: minmax(100px, 1fr) minmax(150px, 2fr) minmax(70px, 0.7fr) minmax(90px, 0.8fr) minmax(150px, 2fr) minmax(100px, 1fr);
        align-items: center;
        gap: 0.5rem;
        padding: 0.8rem 0.5rem;
        border-bottom: 1px solid #e0e0e0;
    }

    .list-header {
        font-weight: bold;
        background-color: #f8f9fa;
        border-bottom: 2px solid #e0e0e0;
        margin-bottom: 0.5rem;
        position: sticky;
        top: 0;
        z-index: 10;
        padding: 0.8rem 0.5rem;
    }
    
    .header-date, .match-date { text-align: left; }
    .header-home-team, .team.home-team { text-align: right; }
    .header-score, .score, .header-xg { text-align: center; }
    .header-away-team, .team.away-team { text-align: left; }
    .header-actions, .match-item-grid > .predict-link-button-small { text-align: center; }


    .match-item {
        background-color: #fff;
    }
    .match-item:nth-child(even) {
    }
    .match-item:hover {
        background-color: #f0f8ff;
    }

    .team {
        font-weight: 500;
    }
    .score {
        font-weight: bold;
    }
    .actual-score {
        font-size: 1.1em;
    }
    .xg-score {
        font-size: 0.9em;
        color: #555;
    }

    .predict-link-button-small {
        padding: 0.3rem 0.6rem;
        font-size: 0.8rem;
        background-color: #5cb85c;
        color: white;
        border-radius: 3px;
        text-decoration: none;
        white-space: nowrap;
        justify-self: center;
    }
    .predict-link-button-small:hover{
        background-color: #4cae4c;
    }

    .pagination-controls {
        margin-top: 2rem;
        display: flex;
        justify-content: center;
        align-items: center;
        gap: 1rem;
    }
    .error-message { color: red; }
</style>
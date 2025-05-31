<script>
    import {onMount} from 'svelte';
    import {apiFetch} from '$lib/api.js';
    import SeasonSelector from '$components/SeasonSelector.svelte';
    import goalsPerSeason from '../../assets/Bar plot of Goals Scored per Match by Season.png'
    import goalsConcededPerSeason from '../../assets/Bar plot of Goals Conceded per Match by Season.png'

    let allSeasons =[];
    let selectedSeasonId = null;

    let teamStatistics = [];
    let isLoadingStats = false;
    let error = null;

    const legendItems = [
        {abbr: 'Sh', full: 'strzały'},
        {abbr: 'SoT', full: 'strzały celne'},
        {abbr: 'FK', full: 'rzuty wolne'},
        {abbr: 'PG', full: 'Gole z rzutów karnych'},
        {abbr: 'PCMP', full: 'podania celne'},
        {abbr: 'PATT', full: 'podania wykonane'},
        {abbr: 'PCMP%', full: 'podania celne w procentach'},
        {abbr: 'CK', full: 'Rzuty rożne'},
        {abbr: 'YC', full: 'Żółte kartki'},
        {abbr: 'RC', full: 'Czerwone kartki'},
        {abbr: 'FC', full: 'Popełnione faule'},
        {abbr: 'PC', full: 'Przewinione rzuty karne (dla przeciwników)'},
        {abbr: 'OG', full: 'Gole samobójcze'},
    ];

    async function fetchSeasons() {
        try{
            const data = await apiFetch('/seasons');
            if(data && data.length > 0) {
                allSeasons = data.map(s => ({id: s.id, name: s.season}))
                selectedSeasonId = 1
            }
        } catch(err) {
            console.error("Error fetching seasons: ", err);
            error = "Nie udało się załadować listy sezonów.";
        }
    }

    async function fetchTeamStatsForSeason(seasonId) {
        if (seasonId === null || seasonId === undefined) {
            teamStatistics = [];
            return;
        }
        isLoadingStats = true;
        error = null;
        try {
            const data = await apiFetch(`/season_stats/season/${seasonId}`);
            teamStatistics = data
        } catch(err) {
            console.error(`Error fetching team stats for season ${seasonId}`, err)
            error = `Nie udało się załadować statystyk dla sezonu ${seasonId}.`;
        } finally {
            isLoadingStats = false;
        }
    }

    onMount(fetchSeasons)

    $: if (selectedSeasonId !== null && selectedSeasonId !== undefined && allSeasons.length > 0) {
        fetchTeamStatsForSeason(selectedSeasonId);
    }

</script>

<svelte:head>
    <title>Statystyki Drużyn - AI Predictor</title>
</svelte:head>

<section class="team-stats-page">
    <h1>Statystyki Drużyn</h1>

    {#if allSeasons.length > 0}
        <SeasonSelector 
            availableSeasons={allSeasons} 
            bind:selectedSeasonId={selectedSeasonId} 
        />
    {:else if !error} <p>Brak dostępnych sezonów do wyświetlenia.</p>
    {:else if error && !isLoadingStats} <p class="error-message">{error}</p>
    {/if}

    <div class="content-grid">
        <div class="stats-table-container">
            {#if isLoadingStats}
                <p>Ładowanie statystyk drużyn...</p>
                {:else if error && teamStatistics.length === 0} <p class="error-message">{error}</p>
                {:else if teamStatistics.length > 0 && selectedSeasonId !== null}
                <h2>Statystyki dla Sezonu ID: {selectedSeasonId}</h2>
                <div class="table-wrapper">
                    <table>
                        <thead>
                            <tr>
                                <th>Drużyna</th>
                                <th>Sh</th>
                                <th>SoT</th>
                                <th>FK</th>
                                <th>PG</th>
                                <th>PCmp</th>
                                <th>PAtt</th>
                                <th>Pcmp%</th>
                                <th>CK</th>
                                <th>YC</th>
                                <th>RC</th>
                                <th>FC</th>
                                <th>PC</th>
                                <th>OG</th>
                            </tr>
                        </thead>
                        <tbody>
                            {#each teamStatistics as teamStat (teamStat.team_id)}
                                <tr>
                                    <td>
                                        <a href="/teams/{teamStat.team_id}" class="team-link">
                                            {teamStat.team.team_name}
                                        </a>
                                    </td>
                                    <td>{teamStat.shots}</td>
                                    <td>{teamStat.shots_on_target}</td>
                                    <td>{teamStat.free_kicks}</td>
                                    <td>{teamStat.penalty_goals}</td>
                                    <td>{teamStat.passes_completed}</td>
                                    <td>{teamStat.passes_attempted}</td>
                                    <td>{teamStat.pass_completion}</td>
                                    <td>{teamStat.corners}</td>
                                    <td>{teamStat.yellow_cards}</td>
                                    <td>{teamStat.red_cards}</td>
                                    <td>{teamStat.fouls_conceded}</td>
                                    <td>{teamStat.penalties_conceded}</td>
                                    <td>{teamStat.own_goals}</td>
                                </tr>
                            {/each}
                        </tbody>
                    </table>
                </div>
            {:else if selectedSeasonId !== null && !isLoadingStats}
                <p>Brak statystyk dla wybranego sezonu.</p>
            {/if}
        </div>
        <aside class="sidebar-right">
            <div class="legend-sidebar card">
                <h2>Legenda Skrótów</h2>
                <ul>
                    {#each legendItems as item (item.abbr)}
                        <li><strong>{item.abbr}</strong>: {item.full}</li>
                    {/each}
                </ul>
            </div>
            {#if !isLoadingStats && teamStatistics.length > 0}
                <div class="visualizations-container card">
                    <h2>Wizualizacje dla Sezonu</h2>
                    <div class="chart-wrapper">
                        <img src={goalsPerSeason} alt="Wykres strzelonych goli">
                        <img src={goalsConcededPerSeason} alt="Wykres straconych goli">
                    </div>
                </div>
            {/if}
        </aside>
    </div>
</section>

<style>
    .team-stats-page h1 {
        color: #333;
        border-bottom: 2px solid #1abc9c;
        padding-bottom: 0.5rem;
        margin-bottom: 1.5rem;
    }
    .content-grid {
        display: grid;
        grid-template-columns: 3fr 1fr;
        gap: 2rem;
    }
    .stats-table-container h2 {
        margin-top: 0;
        font-size: 1.2rem;
        color: #34495e;
    }
    .table-wrapper {
        overflow-x: auto;
    }
    table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.9rem;
        white-space: nowrap;
    }
    th, td {
        text-align: left;
        padding: 0.7rem 0.5rem;
        border-bottom: 1px solid #ecf0f1;
    }
    th {
        background-color: #f2f4f6;
        font-weight: 600;
        color: #555;
        position: sticky;
        top: 0;
        z-index: 1;
    }
    tbody tr:hover {
        background-color: #f9f9f9;
    }
    .team-link {
        color: #007bff;
        text-decoration: none;
        font-weight: 500;
    }
    .team-link:hover {
        text-decoration: underline;
    }
    .legend-sidebar {
        padding: 1.5rem;
        align-self: flex-start;
    }
    .legend-sidebar h2 {
        margin-top: 0;
        font-size: 1.2rem;
        color: #34495e;
        border-bottom: 1px solid #eee;
        padding-bottom: 0.5rem;
        margin-bottom: 1rem;
    }
    .legend-sidebar ul {
        list-style: none;
        padding: 0;
        font-size: 0.9rem;
    }
    .legend-sidebar li {
        margin-bottom: 0.5rem;
    }
    .legend-sidebar strong {
        color: #2c3e50;
    }
    .error-message {
        color: #e74c3c;
        padding: 1rem;
        background-color: #fdeded;
        border: 1px solid #f5c6cb;
        border-radius: 4px;
    }
    .card {
        background: white;
        padding: 1.5rem;
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }

    @media (max-width: 992px) {
        .content-grid {
            grid-template-columns: 1fr;
        }
        .legend-sidebar {
            margin-top: 2rem;
        }
    }


    .sidebar-right {
        display: flex;
        flex-direction: column;
        gap: 2rem;
        align-self: flex-start;
    }

    .visualizations-container h2 {
        margin-top: 0;
        font-size: 1.2rem;
        color: #34495e;
        border-bottom: 1px solid #eee;
        padding-bottom: 0.5rem;
        margin-bottom: 1rem;
    }
    .chart-wrapper {
        margin-bottom: 2rem;
    }

    img {
    max-width: 150%;
    max-height: 150%;
    }
</style>
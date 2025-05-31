<script>
    import { onMount, afterUpdate } from 'svelte';
    import { apiFetch } from '../lib/api.js';
    
    export let seasonId;

    let leagueTable = [];
    let isLoading = false;
    let error = null;

    $: if (seasonId !== null && seasonId !== undefined) {
        fetchTable(seasonId);
    } else {
        leagueTable = [];
    }

    async function fetchTable(currentSeasonId) {
        if (currentSeasonId === null || currentSeasonId === undefined) {
            leagueTable = [];
            return;
        }
        isLoading = true;
        error = null;
        console.log(`LeagueTable: Fetching table for season ID: ${currentSeasonId}`);
        try {
            const data = await apiFetch(`/leagues/${currentSeasonId}`);
            if (Array.isArray(data)) {
                leagueTable = data.sort((a,b) => a.position - b.position);
            } else {
                console.error("LeagueTable: Fetched data is not an array", data);
                leagueTable = [];
            }
        } catch (err) {
            console.error("Error fetching league table:", err);
            error = err.message;
            leagueTable = [];
        } finally {
            isLoading = false;
        }
    }
</script>

{#if seasonId !== null && seasonId !== undefined}
    <h2>Tabela Ligi - Sezon ID: {seasonId}</h2>
    {#if isLoading}
        <p>Ładowanie tabeli ligowej...</p>
    {:else if error}
        <p style="color: red;">Błąd ładowania tabeli: {error}</p>
    {:else if leagueTable.length > 0}
        <table>
            <thead>
                <tr>
                    <th>Pozycja</th>
                    <th>Drużyna</th>
                    <th>Mecze Rozegrane</th>
                    <th>Wygrane</th>
                    <th>Remisy</th>
                    <th>Porażki</th>
                    <th>GF</th>
                    <th>GA</th> <th>GD</th>
                    <th>Punkty</th>
                </tr>
            </thead>
            <tbody>
                {#each leagueTable as teamEntry (teamEntry.id)}
                    <tr>
                        <td>{teamEntry.position}</td>
                        <td>{teamEntry.team?.team_name || 'Brak nazwy'}</td>
                        <td>{teamEntry.games_played}</td>
                        <td>{teamEntry.wins}</td>
                        <td>{teamEntry.draws}</td>
                        <td>{teamEntry.losses}</td>
                        <td>{teamEntry.goals_for}</td>
                        <td>{teamEntry.goals_against}</td>
                        <td>{teamEntry.goals_difference}</td>
                        <td>{teamEntry.points}</td>
                    </tr>
                {/each}
            </tbody>
        </table>
    {:else}
        <p>Brak danych tabeli ligowej dla tego sezonu.</p>
    {/if}
{:else}
    {/if}


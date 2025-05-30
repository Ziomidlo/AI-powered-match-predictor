<script>
    import {onMount} from 'svelte';
    import {apiFetch} from '../lib/api';

    let matches = [];

    onMount(async () => {
        try {
            matches = await apiFetch("/matches/?skip=0&limit=10");
        } catch(error) {
            console.error("Error loading last matches", error);
        }
    });
</script>

<h2>Ostatnie mecze</h2>
<ul>
    {#each matches as match}
        <li>
            {match.match_date} - {match.home_team.team_name} {match.home_goals} : {match.away_goals} {match.away_team.team_name}
        </li>
    {/each}
</ul>
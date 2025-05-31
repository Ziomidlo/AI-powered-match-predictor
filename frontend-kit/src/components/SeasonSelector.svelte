<script>
  export let availableSeasons = [];
  export let selectedSeasonId = null;

</script>


<select bind:value={selectedSeasonId}>
  <option value={null}>Wybierz sezon...</option>
  {#each availableSeasons as season (season.id)}
    <option value={season.id}>{season.name}</option>
  {/each}
</select>

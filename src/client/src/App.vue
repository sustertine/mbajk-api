<script setup lang="ts">
import ModeToggle from "@/components/ModeToggle.vue";
import SideBar from "@/components/SideBar.vue";
import Button from "@/components/ui/button/Button.vue";
import {onMounted, ref} from "vue";
import L from 'leaflet'
import 'leaflet/dist/leaflet.css'

const VITE_APP_BACKEND_URL = import.meta.env.VITE_APP_BACKEND_URL;

const setSelectedStation = async (value: any) => {
  selectedStation.value = value;
  lat.value = value.lat;
  lng.value = value.lng;
  map.setView([lat.value, lng.value], 16);

  const data = {
    station_name: value.name,
    temperature: 25.1,
    relative_humidity: 45,
    dew_point: 12.4,
    apparent_temperature: 24.7,
    precipitation_probability: 0.0,
    rain: 0.0,
    surface_pressure: 984.3,
    bike_stands: 22,
    available_bike_stands: 8
  };

  const response = await fetch(`${VITE_APP_BACKEND_URL}/api/mbajk/predict`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify(data)
  });

  const result = await response.json();
  predictedBikeSlots.value = result.prediction;
}

const selectedStation = ref('');
const lat = ref(46.5547);
const lng = ref(15.6459);
const predictedBikeSlots = ref([0, 0, 0, 0, 0, 0, 0]);
let map: any;

onMounted(async () => {
  map = L.map('map').setView([lat.value, lng.value], 13);

  L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    maxZoom: 19,
  }).addTo(map);

  const response = await fetch(`${VITE_APP_BACKEND_URL}/api/mbajk/stations/info`);
  const data = await response.json();

  data.forEach((item: any) => {
    L.marker([item.lat, item.lng]).addTo(map).bindPopup(item.name);
  });
})
</script>

<template>
  <div>
<!--    <ModeToggle/>-->
  </div>
  <div class="flex w-full h-screen">
    <div class="w-1/5">
      <SideBar @updateValue="setSelectedStation" :predictedBikeSlots="predictedBikeSlots"/>
    </div>
    <div class="w-4/5 p-3">
      <div id="map" class="w-full h-full"></div>
    </div>
  </div>
</template>

<style scoped>
</style>
<script setup lang="ts">
import {defineComponent, onMounted, ref, withDefaults} from 'vue'
import {Check, ChevronsUpDown} from 'lucide-vue-next'
import {cn} from '@/lib/utils'
import {Button} from '@/components/ui/button'
import {
  Command,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList,
} from '@/components/ui/command'
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from '@/components/ui/popover'
import {FontAwesomeIcon} from "@fortawesome/vue-fontawesome";
import {faBicycle} from "@fortawesome/free-solid-svg-icons";
import Card from "@/components/ui/card/Card.vue";
import CardHeader from "@/components/ui/card/CardHeader.vue";
import CardTitle from "@/components/ui/card/CardTitle.vue";
import CardContent from "@/components/ui/card/CardContent.vue";
import ModeToggle from "@/components/ModeToggle.vue";

const items = ref<Array<{ value: string, label: string, lat: number, lng: number }>>([])

const open = ref(false)
const value = ref('')

const times = Array.from({length: 7}, (_, i) => {
  const date = new Date();
  date.setHours(date.getHours() + i + 1);
  return date.toLocaleTimeString([], {hour: '2-digit', minute: '2-digit', hour12: false}); // Display the time in a 24-hour format
});

onMounted(async () => {
  const response = await fetch('http://localhost:8000/api/mbajk/stations/info')
  const data = await response.json()
  items.value = data.map(item => ({value: item.name, label: item.name, lat: item.lat, lng: item.lng}))
})

const props = defineProps({
  predictedBikeSlots: Array,
});

const emit = defineEmits(['updateValue'])
</script>

<template>
  <div class="flex flex-col justify-start w-full h-full p-3">
    <div class="w-100 align-end text-end mb-4">
      <ModeToggle/>
    </div>
    <Popover v-model:open="open">
      <PopoverTrigger as-child>
        <Button
            variant="outline"
            role="combobox"
            :aria-expanded="open"
            class="w-full justify-between"
        >
          {{ value || 'Select station' }}
          <ChevronsUpDown class="ml-2 h-4 w-4 shrink-0 opacity-50"/>
        </Button>
      </PopoverTrigger>
      <PopoverContent class="w-full p-0">
        <Command>
          <CommandInput class="h-9" placeholder="Search station..."/>
          <CommandEmpty>No stations found.</CommandEmpty>
          <CommandList>
            <CommandGroup>
              <CommandItem
                  v-for="item in items"
                  :key="item.value"
                  :value="item.value"
                  @select="(ev) => {
                  if (typeof ev.detail.value === 'string') {
                    value = ev.detail.value
                    emit('updateValue', item) // Emit the entire item
                  }
                  open = false
                }"
              >
                {{ item.label }}
                <Check
                    :class="cn(
                    'ml-auto h-4 w-4',
                    value === item.value ? 'opacity-100' : 'opacity-0',
                  )"
                />
              </CommandItem>
            </CommandGroup>
          </CommandList>
        </Command>
      </PopoverContent>
    </Popover>
    <Card class="text-center mt-4" v-if="value">
      <CardHeader class="mt-4 md text-center font-bold">
        <CardTitle>{{ value }}</CardTitle>
        <CardTitle class="text-4xl text-center mt-4">
          123
        </CardTitle>
        <div class="mt-2 text-center">
          <FontAwesomeIcon :icon="faBicycle" class="fa-2x"/>
        </div>
      </CardHeader>
      <CardContent class="flex justify-around mt-4">
        <div v-for="(time, index) in times" :key="index" class="text-center">
          <div class="bike-text">{{ time }}</div>
          <div class="bike-text font-bold">{{ predictedBikeSlots[index] }}</div>
          <FontAwesomeIcon :icon="faBicycle" class="fa-1x"/>
        </div>
      </CardContent>
    </Card>
  </div>
</template>
import axios from 'axios'

const BASE = import.meta.env.VITE_API_URL ?? '/api'

export const analyzeSingle = (text) =>
  axios.post(`${BASE}/analyze/single`, { text }).then(r => r.data)

export const analyzeConversation = (messages) =>
  axios.post(`${BASE}/analyze/conversation`, { messages }).then(r => r.data)

export const runStressTest = (additional_samples = null) =>
  axios.post(`${BASE}/evaluate/stress-test`, { additional_samples }).then(r => r.data)

export const getResources = () =>
  axios.get(`${BASE}/resources`).then(r => r.data)

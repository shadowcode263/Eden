"use client"

import { useEffect, useState, useCallback } from "react"
import { Zap } from "lucide-react"
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts"
import { useWebSocket } from "@/hooks/use-websocket" // Import useWebSocket

interface NeuronFiringDisplayProps {
  networkId: number // Now accepts networkId as a prop
}

interface RetrievalData {
  query_text: string
  confidence_score: number
  steps: Array<{
    // Standardized to 'steps' for consistency with WebSocket data
    step: number
    state: number[]
    weights: number[]
  }>
  retrieved_hash: string // Standardized to 'retrieved_hash' for consistency
  retrieved_text: string
}

interface FiringNeuron {
  index: number
  value: number
  intensity: number
}

export function NeuronFiringDisplay({ networkId }: NeuronFiringDisplayProps) {
  const [firingNeurons, setFiringNeurons] = useState<FiringNeuron[]>([])
  const [displayThreshold, setDisplayThreshold] = useState(0.001)
  const [retrievalData, setRetrievalData] = useState<RetrievalData | null>(null)
  const [loading, setLoading] = useState(true)
  const [chartData, setChartData] = useState<any[]>([])

  // Use the WebSocket hook to listen for updates
  const { lastMessage } = useWebSocket(networkId)

  const fetchFiringData = useCallback(async () => {
    if (!networkId) {
      setLoading(false)
      return
    }
    setLoading(true)
    try {
      const response = await fetch(`http://localhost:8000/api/networks/last-retrieval-firing-data/${networkId}/`)
      if (response.ok) {
        const apiData = await response.json()
        // Normalize API data to match the 'steps' and 'retrieved_hash' structure
        const normalizedData: RetrievalData = {
          query_text: apiData.query_text,
          confidence_score: apiData.confidence_score,
          steps: apiData.retrieval_steps_data, // Map from backend's retrieval_steps_data
          retrieved_hash: apiData.retrieved_pattern_hash, // Map from backend's retrieved_pattern_hash
          retrieved_text: apiData.retrieved_text,
        }
        setRetrievalData(normalizedData)
        console.log("NeuronFiringDisplay: Fetched and normalized retrieval data", normalizedData)
      } else if (response.status === 404) {
        setRetrievalData(null) // No retrieval data yet
      } else {
        console.error("Failed to fetch neuron firing data:", response.statusText)
        setRetrievalData(null)
      }
    } catch (error) {
      console.error("Error fetching neuron firing data:", error)
      setRetrievalData(null)
    } finally {
      setLoading(false)
    }
  }, [networkId])

  useEffect(() => {
    fetchFiringData()
  }, [networkId, fetchFiringData])

  // Listen for WebSocket updates to re-fetch data
  useEffect(() => {
    if (lastMessage && lastMessage.event_type === "pattern_retrieved") {
      console.log("NeuronFiringDisplay: WebSocket update received, processing data.", lastMessage.data)
      const newRetrievalData = lastMessage.data as RetrievalData
      // Only update if the new retrieval is different from the current one
      if (
        retrievalData?.retrieved_hash !== newRetrievalData.retrieved_hash ||
        retrievalData?.steps?.length !== newRetrievalData.steps?.length
      ) {
        setRetrievalData(newRetrievalData)
        setLoading(false) // Data received via WebSocket
      }
    }
  }, [lastMessage, retrievalData]) // Include retrievalData to compare against current state

  useEffect(() => {
    if (retrievalData?.steps?.length > 0) {
      const finalStep = retrievalData.steps[retrievalData.steps.length - 1]
      console.log("NeuronFiringDisplay: Final step state values (first 10)", finalStep.state.slice(0, 10))
      console.log("NeuronFiringDisplay: Max state value in final step:", Math.max(...finalStep.state))
      console.log("NeuronFiringDisplay: Min state value in final step:", Math.min(...finalStep.state))

      const firing = finalStep.state
        .map((value, index) => ({
          index,
          value,
          intensity: Math.min(value * 100, 100),
        }))
        .filter((neuron) => neuron.value > displayThreshold)
        .sort((a, b) => b.value - a.value)
        .slice(0, 5) // Changed to show only 5 top firing neurons

      console.log("NeuronFiringDisplay: Calculated firing neurons (after threshold)", firing)
      setFiringNeurons(firing)
    } else {
      setFiringNeurons([]) // Clear neurons if no data or steps
      console.log("NeuronFiringDisplay: Chart Data (cleared)", []) // Log when cleared
    }
  }, [retrievalData, displayThreshold])

  // chartData is now calculated and logged inside the useEffect,
  // but it's also used in the JSX. To avoid re-calculating it on every render
  // caused by currentStep, we should memoize it or make it a state.
  // Let's make it a state, updated when firingNeurons changes.
  useEffect(() => {
    const newChartData = firingNeurons.map((neuron) => ({
      name: `N${neuron.index}`,
      value: neuron.value,
    }))
    setChartData(newChartData)
  }, [firingNeurons])

  if (loading) {
    return (
      <div className="text-center py-8 text-gray-500">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600 mx-auto mb-4"></div>
        <p>Loading neuron firing data...</p>
      </div>
    )
  }

  if (!retrievalData || firingNeurons.length === 0) {
    return (
      <div className="text-center py-8 text-gray-500">
        <Zap className="w-12 h-12 mx-auto mb-4 text-gray-300" />
        <p>No neuron firing data available</p>
        <p className="text-sm">Perform a retrieval to see which neurons fire</p>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Firing Pattern Visualization - Graph */}
      <div>
        <h3 className="font-medium text-gray-900 mb-3">Firing Pattern (Activation Values)</h3>
        <div className="bg-gray-50 p-4 rounded-lg border border-gray-200 h-[300px]">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={chartData} margin={{ top: 5, right: 10, left: 10, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" />
              <XAxis
                dataKey="name"
                tick={{ fill: "#6b7280", fontSize: 10 }}
                axisLine={{ stroke: "#d1d5db" }}
                tickLine={{ stroke: "#d1d5db" }}
              />
              <YAxis
                tick={{ fill: "#6b7280", fontSize: 10 }}
                axisLine={{ stroke: "#d1d5db" }}
                tickLine={{ stroke: "#d1d5db" }}
                domain={[0, Math.max(...firingNeurons.map((n) => n.value), 0.1)]} // Dynamic Y-axis max
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: "rgba(255, 255, 255, 0.9)",
                  border: "1px solid #e0e0e0",
                  borderRadius: "8px",
                  fontSize: "12px",
                }}
                labelStyle={{ color: "#374151", fontWeight: "bold" }}
                itemStyle={{ color: "#4b5563" }}
                formatter={(value: number) => [`${value.toFixed(4)}`, "Activation"]}
              />
              <Bar dataKey="value" fill="#00ff00" radius={[4, 4, 0, 0]} /> {/* Lime green bars */}
            </BarChart>
          </ResponsiveContainer>
        </div>
        <div className="mt-2 text-xs text-gray-500 text-center">
          Bar height indicates neuron activation value. Only neurons above threshold (
          {(displayThreshold * 100).toFixed(1)}%) are shown.
        </div>
      </div>
    </div>
  )
}

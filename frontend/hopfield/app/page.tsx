"use client"

import { useState, useEffect, useCallback } from "react" // Import useCallback
import { BrainVisualization } from "@/components/brain-visualization"
import { CreateNetworkModal } from "@/components/create-network-modal"
import { BrainControlPanel } from "@/components/brain-control-panel"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Brain, Plus, Settings, GitBranch, Zap } from "lucide-react"
import { NeuronFiringDisplay } from "@/components/neuron-firing-display"
import { Badge } from "@/components/ui/badge" // Import Badge

interface Network {
  id: number
  name: string
  embedding_dim: number
  beta: number
  learning_rate: number
  total_patterns: number
  total_retrievals: number
  merkle_root: string | null
  created_at: string
}

interface RetrievalSummaryData {
  query_text: string
  confidence_score: number
  steps_count: number
  retrieved_hash: string
  retrieved_text: string
}

export default function HomePage() {
  const [networks, setNetworks] = useState<Network[]>([])
  const [selectedNetwork, setSelectedNetwork] = useState<Network | null>(null)
  const [showCreateModal, setShowCreateModal] = useState(false)
  const [loading, setLoading] = useState(true)
  const [retrievalSummaryData, setRetrievalSummaryData] = useState<RetrievalSummaryData | null>(null)

  useEffect(() => {
    fetchNetworks()
  }, [])

  const fetchNetworks = async () => {
    try {
      const response = await fetch("http://localhost:8000/api/networks/")
      const data = await response.json()
      setNetworks(data)
      if (data.length > 0 && !selectedNetwork) {
        setSelectedNetwork(data[0])
      } else if (selectedNetwork) {
        const updatedSelected = data.find((n) => n.id === selectedNetwork.id)
        if (updatedSelected) {
          setSelectedNetwork(updatedSelected)
        } else if (data.length > 0) {
          setSelectedNetwork(data[0])
        } else {
          setSelectedNetwork(null)
        }
      }
    } catch (error) {
      console.error("Error fetching networks:", error)
    } finally {
      setLoading(false)
    }
  }

  const handleNetworkCreated = (network: Network) => {
    setNetworks((prev) => [network, ...prev])
    setSelectedNetwork(network)
    setShowCreateModal(false)
  }

  const handleNetworkUpdated = (updatedNetwork: Network) => {
    setNetworks((prev) => prev.map((n) => (n.id === updatedNetwork.id ? updatedNetwork : n)))
    setSelectedNetwork(updatedNetwork)
  }

  // Memoize the callback to prevent unnecessary re-renders of BrainVisualization
  const handleRetrievalSummaryUpdate = useCallback((data: RetrievalSummaryData) => {
    setRetrievalSummaryData(data)
  }, [])

  if (loading) {
    return (
      <div className="min-h-screen bg-white flex items-center justify-center">
        <div className="text-center">
          <Brain className="w-20 h-20 text-blue-600 animate-pulse mx-auto mb-6" />
          <p className="text-xl text-gray-700">Initializing Neural Networks...</p>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-white text-foreground">
      {/* Header */}
      <header className="bg-white border-b border-gray-200 sticky top-0 z-50 shadow-sm">
        <div className="w-full px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <div className="p-2 bg-blue-100 rounded-lg">
                <Brain className="w-8 h-8 text-blue-600" />
              </div>
              <div>
                <h1 className="text-2xl font-bold text-gray-900">Hopfield Brain</h1>
                <p className="text-sm text-gray-600">Adaptive Neural Memory System</p>
              </div>
            </div>

            <div className="flex items-center space-x-4">
              <select
                value={selectedNetwork?.id || ""}
                onChange={(e) => {
                  const network = networks.find((n) => n.id === Number.parseInt(e.target.value))
                  setSelectedNetwork(network || null)
                  setRetrievalSummaryData(null) // Clear retrieval summary when network changes
                }}
                className="px-4 py-2 border border-gray-300 rounded-lg text-gray-900 focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              >
                <option value="">Select Network</option>
                {networks.map((network) => (
                  <option key={network.id} value={network.id}>
                    {network.name} ({network.total_patterns} patterns)
                  </option>
                ))}
              </select>

              <Button onClick={() => setShowCreateModal(true)} className="bg-blue-600 hover:bg-blue-700">
                <Plus className="w-4 h-4 mr-2" />
                New Network
              </Button>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content Area */}
      <main className="w-full py-8 px-6">
        {!selectedNetwork ? (
          <section className="text-center py-20">
            <div className="p-8 bg-gray-50 rounded-xl border border-gray-200 max-w-md mx-auto shadow-lg">
              <Brain className="w-24 h-24 text-gray-400 mx-auto mb-6" />
              <h2 className="text-2xl font-bold text-gray-900 mb-4">No Network Selected</h2>
              <p className="text-gray-600 mb-8">Create or select a neural network to get started</p>
              <Button onClick={() => setShowCreateModal(true)} size="lg" className="bg-blue-600 hover:bg-blue-700">
                <Plus className="w-5 h-5 mr-2" />
                Create Your First Network
              </Button>
            </div>
          </section>
        ) : (
          <div className="grid grid-cols-1 lg:grid-cols-10 gap-6 h-[calc(100vh-160px)]">
            {/* Left Column: Neural Network Visualization */}
            <section className="lg:col-span-7 h-full flex flex-col">
              <Card className="shadow-lg border-gray-200 bg-white text-foreground h-full flex flex-col">
                <CardHeader className="pb-4">
                  <div className="flex items-center justify-between flex-wrap gap-4">
                    <div className="flex items-center">
                      <Brain className="w-6 h-6 mr-3 text-blue-600" />
                      <div>
                        <CardTitle className="text-xl text-gray-900">Neural Network Visualization</CardTitle>
                        <CardDescription className="text-gray-600">
                          Real-time 3D visualization of {selectedNetwork.name}
                        </CardDescription>
                      </div>
                    </div>
                    <div className="flex items-center space-x-3 flex-wrap gap-2">
                      {retrievalSummaryData?.retrieved_text && (
                        <Badge variant="outline" className="text-purple-700 border-purple-300 bg-purple-100">
                          Retrieved: "{retrievalSummaryData.retrieved_text}"
                        </Badge>
                      )}
                      {retrievalSummaryData?.confidence_score !== undefined && (
                        <Badge variant="outline" className="text-green-700 border-green-300 bg-green-100">
                          Confidence: {(retrievalSummaryData.confidence_score * 100).toFixed(1)}%
                        </Badge>
                      )}
                      {retrievalSummaryData?.steps_count !== undefined && (
                        <Badge variant="outline" className="text-orange-700 border-orange-300 bg-orange-100">
                          Steps: {retrievalSummaryData.steps_count}
                        </Badge>
                      )}
                      <Badge
                        variant="outline"
                        className="border-gray-300 text-gray-700 bg-gray-100 hover:bg-gray-100 hover:text-gray-900"
                      >
                        <GitBranch className="w-4 h-4 mr-2" />
                        {selectedNetwork.merkle_root ? `${selectedNetwork.merkle_root}` : "N/A"}
                      </Badge>
                      <div className="px-3 py-1 bg-blue-100 rounded-full text-blue-700 text-sm font-medium border border-blue-200">
                        {selectedNetwork.total_patterns} patterns
                      </div>
                    </div>
                  </div>
                </CardHeader>
                <CardContent className="flex-grow p-0">
                  <BrainVisualization
                    networkId={selectedNetwork.id}
                    embeddingDim={selectedNetwork.embedding_dim}
                    onRetrievalSummaryUpdate={handleRetrievalSummaryUpdate}
                  />
                </CardContent>
              </Card>
            </section>
            {/* Right Column: Controls and Stats */}
            <section className="lg:col-span-3 space-y-6 overflow-y-auto pr-2 custom-scrollbar">
              {/* Brain Control Panel */}
              <Card className="shadow-lg border-gray-200 bg-white text-foreground">
                <CardHeader>
                  <CardTitle className="flex items-center text-gray-900">
                    <Settings className="w-5 h-5 mr-2 text-blue-600" />
                    Brain Control Panel
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <BrainControlPanel network={selectedNetwork} onNetworkUpdated={handleNetworkUpdated} />
                </CardContent>
              </Card>
              {/* Neuron Firing Display */}
              <Card className="shadow-lg border-gray-200 bg-white text-foreground">
                <CardHeader className="pb-4">
                  <CardTitle className="flex items-center text-gray-900">
                    <Zap className="w-5 h-5 mr-2 text-red-600" />
                    Neuron Firing Pattern
                  </CardTitle>
                  <CardDescription className="text-gray-600">
                    Neurons that fired during the last retrieval
                  </CardDescription>
                </CardHeader>
                <CardContent className="w-full">
                  <NeuronFiringDisplay networkId={selectedNetwork.id} />
                </CardContent>
              </Card>
            </section>
          </div>
        )}
      </main>

      {/* Modals */}
      <CreateNetworkModal
        isOpen={showCreateModal}
        onClose={() => setShowCreateModal(false)}
        onNetworkCreated={handleNetworkCreated}
      />
    </div>
  )
}

"use client"

import { useState, useEffect } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Badge } from "@/components/ui/badge"
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog"
import { Brain, Plus, Settings, Info, BookOpen, Gamepad2 } from "lucide-react"
import { toast } from "@/hooks/use-toast"
import { Toaster } from "@/components/ui/toaster"
import { BrainVisualization } from "@/components/brain-visualization"
import { TextTrainingPanel } from "@/components/text-training-panel"
import { GameTrainingPanel } from "@/components/game-training-panel"
import { NetworkDetails } from "@/components/network-details"
import { CreateNetworkModal } from "@/components/create-network-modal"

interface Network {
  id: number
  name: string
  description: string
  is_active: boolean
  created_at: string
}

export default function Home() {
  const [networks, setNetworks] = useState<Network[]>([])
  const [activeNetwork, setActiveNetwork] = useState<Network | null>(null)
  const [isCreateModalOpen, setIsCreateModalOpen] = useState(false)
  const [isNetworkDetailsOpen, setIsNetworkDetailsOpen] = useState(false)
  const [refreshTrigger, setRefreshTrigger] = useState(0)

  useEffect(() => {
    fetchNetworks()
  }, [refreshTrigger])

  const fetchNetworks = async () => {
    try {
      const response = await fetch("http://localhost:8000/api/brain/networks/")
      if (response.ok) {
        const data = await response.json()
        setNetworks(data)
        const active = data.find((n: Network) => n.is_active)
        setActiveNetwork(active || null)
      }
    } catch (error) {
      console.error("Failed to fetch networks:", error)
    }
  }

  const handleSetActiveNetwork = async (networkId: number) => {
    try {
      const response = await fetch(`http://localhost:8000/api/brain/networks/${networkId}/set-active/`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ action: "set_active" }),
      })
      if (response.ok) {
        toast({ title: "Network Activated", description: "Network is now active." })
        setRefreshTrigger((prev) => prev + 1)
      } else {
        const errorData = await response.json()
        throw new Error(errorData.error || "Failed to activate network")
      }
    } catch (error) {
      console.error("Error activating network:", error)
      toast({
        title: "Error",
        description: error instanceof Error ? error.message : "Failed to activate network.",
        variant: "destructive",
      })
    }
  }

  const handleRefresh = () => {
    setRefreshTrigger((prev) => prev + 1)
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950">
      {/* Header */}
      <header className="border-b border-slate-800/50 bg-slate-900/80 backdrop-blur-xl supports-[backdrop-filter]:bg-slate-900/80 shadow-2xl">
        <div className="px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className="relative">
                <Brain className="w-8 h-8 text-blue-400" />
                <div className="absolute inset-0 w-8 h-8 text-blue-400 animate-pulse opacity-50"></div>
              </div>
              <div>
                <h1 className="text-2xl font-bold bg-gradient-to-r from-blue-400 to-cyan-400 bg-clip-text text-transparent">
                  STAG Brain Network
                </h1>
                <p className="text-sm text-slate-400 font-mono">Spatio-Temporal Associative Graph Interface</p>
              </div>
            </div>
            <div className="flex items-center space-x-4">
              <div className="flex items-center space-x-2">
                <span className="text-sm font-medium text-slate-300">Active Network:</span>
                <Select
                  value={activeNetwork?.id.toString() || ""}
                  onValueChange={(value) => handleSetActiveNetwork(Number(value))}
                >
                  <SelectTrigger className="w-48 bg-slate-800/50 border-slate-700 text-slate-200 hover:bg-slate-700/50 transition-all duration-200">
                    <SelectValue placeholder="Select network">
                      {activeNetwork ? (
                        <div className="flex items-center space-x-2">
                          <span>{activeNetwork.name}</span>
                          <Badge
                            variant="secondary"
                            className="text-xs bg-green-500/20 text-green-400 border-green-500/30"
                          >
                            Active
                          </Badge>
                        </div>
                      ) : (
                        "No active network"
                      )}
                    </SelectValue>
                  </SelectTrigger>
                  <SelectContent className="bg-slate-800 border-slate-700">
                    {networks.map((network) => (
                      <SelectItem
                        key={network.id}
                        value={network.id.toString()}
                        className="text-slate-200 hover:bg-slate-700"
                      >
                        <div className="flex items-center justify-between w-full">
                          <span>{network.name}</span>
                          {network.is_active && (
                            <Badge
                              variant="secondary"
                              className="ml-2 text-xs bg-green-500/20 text-green-400 border-green-500/30"
                            >
                              Active
                            </Badge>
                          )}
                        </div>
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              {/* Network Details Dialog */}
              <Dialog open={isNetworkDetailsOpen} onOpenChange={setIsNetworkDetailsOpen}>
                <DialogTrigger asChild>
                  <Button
                    variant="outline"
                    size="sm"
                    disabled={!activeNetwork}
                    className="bg-slate-800/50 border-slate-700 text-slate-200 hover:bg-slate-700/50"
                  >
                    <Info className="w-4 h-4 mr-2" />
                    Network Details
                  </Button>
                </DialogTrigger>
                <DialogContent className="max-w-4xl max-h-[80vh] overflow-y-auto bg-slate-900 border-slate-700">
                  <DialogHeader>
                    <DialogTitle className="flex items-center gap-2 text-slate-200">
                      <Settings className="w-5 h-5 text-blue-400" />
                      Network Details
                      {activeNetwork && (
                        <Badge variant="secondary" className="bg-green-500/20 text-green-400 border-green-500/30">
                          Active
                        </Badge>
                      )}
                    </DialogTitle>
                  </DialogHeader>
                  <NetworkDetails networkId={activeNetwork?.id || null} onRefresh={handleRefresh} />
                </DialogContent>
              </Dialog>

              <Button
                onClick={() => setIsCreateModalOpen(true)}
                size="sm"
                className="bg-blue-600 hover:bg-blue-700 text-white shadow-lg"
              >
                <Plus className="w-4 h-4 mr-2" />
                Create Network
              </Button>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="px-6 py-6 space-y-6">
        {/* Two Panel Layout - Equal Heights */}
        {activeNetwork ? (
          <div className="flex h-[85vh] gap-6">
            {/* Left Panel - Brain Visualization (50% width) */}
            <div className="w-1/2 h-full">
              <BrainVisualization networkId={activeNetwork.id} />
            </div>

            {/* Right Panel - Training Interface (50% width) */}
            <div className="w-1/2 h-full">
              <div className="h-full flex flex-col space-y-4">
                {/* Text Training Panel */}
                <Card className="flex-1 bg-gradient-to-br from-slate-900/90 to-slate-800/90 border-slate-700/50 shadow-2xl backdrop-blur-sm">
                  <CardContent className="flex-1 overflow-y-auto">
                    <TextTrainingPanel networkId={activeNetwork.id} onAction={handleRefresh} />
                  </CardContent>
                </Card>

                {/* Game Training Panel */}
                <Card className="flex-1 bg-gradient-to-br from-slate-800/90 to-slate-900/90 border-slate-700/50 shadow-2xl backdrop-blur-sm">
                  <CardContent className="flex-1 overflow-y-auto">
                    <GameTrainingPanel networkId={activeNetwork.id} />
                  </CardContent>
                </Card>
              </div>
            </div>
          </div>
        ) : (
          <Card className="w-full h-[85vh] bg-slate-900/80 border-slate-800 shadow-2xl backdrop-blur-sm">
            <CardContent className="flex flex-col items-center justify-center h-full">
              <div className="relative mb-6">
                <Brain className="w-16 h-16 text-slate-400" />
                <div className="absolute inset-0 w-16 h-16 text-blue-400 animate-pulse opacity-50"></div>
              </div>
              <h3 className="text-lg font-semibold mb-2 text-slate-200">No Active Network</h3>
              <p className="text-slate-400 text-center mb-4 font-mono">
                Create a new network or select an existing one to begin neural processing.
              </p>
              <Button
                onClick={() => setIsCreateModalOpen(true)}
                className="bg-blue-600 hover:bg-blue-700 text-white shadow-lg"
              >
                <Plus className="w-4 h-4 mr-2" />
                Create Network
              </Button>
            </CardContent>
          </Card>
        )}
      </main>

      <CreateNetworkModal
        isOpen={isCreateModalOpen}
        onClose={() => setIsCreateModalOpen(false)}
        onNetworkCreated={handleRefresh}
      />
      <Toaster />
    </div>
  )
}

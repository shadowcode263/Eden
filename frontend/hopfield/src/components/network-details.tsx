"use client"

import { useState, useEffect } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "@/components/ui/accordion"
import { Activity, Brain, Settings, Zap, Target, Layers } from 'lucide-react'
import { toast } from "@/hooks/use-toast"

interface NetworkDetailsProps {
  networkId: number | null
  onRefresh: () => void
}

interface NetworkStats {
  total_nodes: number
  total_edges: number
  active_nodes: number
  learning_rate: number
  total_patterns: number
  total_retrievals: number
}

interface NetworkParameters {
  sdr_dimensionality: number
  sdr_sparsity: number
  winner_learning_rate: number
  neighbor_learning_rate: number
  error_decay_rate: number
  max_edge_age: number
  n_iter_before_neuron_added: number
  max_nodes: number
  cells_per_column: number
  initial_permanence: number
  connected_permanence: number
  permanence_increment: number
  permanence_decrement: number
  activation_threshold: number
  rl_learning_rate: number
  rl_discount_factor: number
  rl_exploration_rate: number
}

export function NetworkDetails({ networkId, onRefresh }: NetworkDetailsProps) {
  const [stats, setStats] = useState<NetworkStats | null>(null)
  const [parameters, setParameters] = useState<NetworkParameters | null>(null)
  const [isLoading, setIsLoading] = useState(false)

  useEffect(() => {
    if (networkId) {
      fetchNetworkData()
    }
  }, [networkId])

  const fetchNetworkData = async () => {
    if (!networkId) return

    setIsLoading(true)
    try {
      // Fetch network details
      const networkResponse = await fetch(`http://localhost:8000/api/brain/networks/${networkId}/`)
      const networkData = await networkResponse.json()
      setParameters(networkData)

      // Fetch network stats from graph state
      const stateResponse = await fetch(`http://localhost:8000/api/brain/state/?network_id=${networkId}`)
      const stateData = await stateResponse.json()
      setStats({
        total_nodes: stateData.nodes?.length || 0,
        total_edges: stateData.links?.length || 0,
        active_nodes: stateData.nodes?.filter((n: any) => n.active)?.length || 0,
        learning_rate: networkData.winner_learning_rate || 0,
        total_patterns: stateData.total_patterns || 0,
        total_retrievals: stateData.total_retrievals || 0,
      })
    } catch (error) {
      console.error("Error fetching network data:", error)
      toast({
        title: "Error",
        description: "Failed to fetch network details.",
        variant: "destructive",
      })
    } finally {
      setIsLoading(false)
    }
  }

  if (!networkId) {
    return (
      <Card>
        <CardContent className="flex items-center justify-center py-8">
          <p className="text-muted-foreground">No network selected</p>
        </CardContent>
      </Card>
    )
  }

  return (
    <div className="space-y-4">
      {/* Network Statistics */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Activity className="w-5 h-5 text-primary" />
            Network Statistics
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
            <div className="text-center">
              <div className="text-2xl font-bold text-primary">{stats?.total_nodes || 0}</div>
              <div className="text-sm text-muted-foreground">Total Nodes</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-primary">{stats?.total_edges || 0}</div>
              <div className="text-sm text-muted-foreground">Total Edges</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-primary">{stats?.active_nodes || 0}</div>
              <div className="text-sm text-muted-foreground">Active Nodes</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-primary">{stats?.total_patterns || 0}</div>
              <div className="text-sm text-muted-foreground">Patterns</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-primary">{stats?.total_retrievals || 0}</div>
              <div className="text-sm text-muted-foreground">Retrievals</div>
            </div>
            <div className="text-center">
              <Button onClick={fetchNetworkData} variant="outline" size="sm" disabled={isLoading}>
                <Activity className="w-4 h-4 mr-2" />
                Refresh
              </Button>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Network Parameters */}
      {parameters && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Settings className="w-5 h-5 text-primary" />
              Network Parameters
            </CardTitle>
          </CardHeader>
          <CardContent>
            <Accordion type="single" collapsible className="w-full">
              <AccordionItem value="core">
                <AccordionTrigger className="flex items-center gap-2">
                  <Brain className="w-4 h-4" />
                  Core Parameters
                </AccordionTrigger>
                <AccordionContent>
                  <div className="grid grid-cols-2 gap-4 text-sm">
                    <div>
                      <span className="font-medium">SDR Dimensionality:</span>
                      <Badge variant="secondary" className="ml-2">{parameters.sdr_dimensionality}</Badge>
                    </div>
                    <div>
                      <span className="font-medium">SDR Sparsity:</span>
                      <Badge variant="secondary" className="ml-2">{parameters.sdr_sparsity}</Badge>
                    </div>
                    <div>
                      <span className="font-medium">Max Nodes:</span>
                      <Badge variant="secondary" className="ml-2">{parameters.max_nodes}</Badge>
                    </div>
                    <div>
                      <span className="font-medium">Max Edge Age:</span>
                      <Badge variant="secondary" className="ml-2">{parameters.max_edge_age}</Badge>
                    </div>
                  </div>
                </AccordionContent>
              </AccordionItem>

              <AccordionItem value="gng">
                <AccordionTrigger className="flex items-center gap-2">
                  <Zap className="w-4 h-4" />
                  GNG Learning
                </AccordionTrigger>
                <AccordionContent>
                  <div className="grid grid-cols-2 gap-4 text-sm">
                    <div>
                      <span className="font-medium">Winner Learning Rate:</span>
                      <Badge variant="secondary" className="ml-2">{parameters.winner_learning_rate}</Badge>
                    </div>
                    <div>
                      <span className="font-medium">Neighbor Learning Rate:</span>
                      <Badge variant="secondary" className="ml-2">{parameters.neighbor_learning_rate}</Badge>
                    </div>
                    <div>
                      <span className="font-medium">Error Decay Rate:</span>
                      <Badge variant="secondary" className="ml-2">{parameters.error_decay_rate}</Badge>
                    </div>
                    <div>
                      <span className="font-medium">Iterations Before Add:</span>
                      <Badge variant="secondary" className="ml-2">{parameters.n_iter_before_neuron_added}</Badge>
                    </div>
                  </div>
                </AccordionContent>
              </AccordionItem>

              <AccordionItem value="htm">
                <AccordionTrigger className="flex items-center gap-2">
                  <Layers className="w-4 h-4" />
                  HTM Learning
                </AccordionTrigger>
                <AccordionContent>
                  <div className="grid grid-cols-2 gap-4 text-sm">
                    <div>
                      <span className="font-medium">Cells per Column:</span>
                      <Badge variant="secondary" className="ml-2">{parameters.cells_per_column}</Badge>
                    </div>
                    <div>
                      <span className="font-medium">Initial Permanence:</span>
                      <Badge variant="secondary" className="ml-2">{parameters.initial_permanence}</Badge>
                    </div>
                    <div>
                      <span className="font-medium">Connected Permanence:</span>
                      <Badge variant="secondary" className="ml-2">{parameters.connected_permanence}</Badge>
                    </div>
                    <div>
                      <span className="font-medium">Activation Threshold:</span>
                      <Badge variant="secondary" className="ml-2">{parameters.activation_threshold}</Badge>
                    </div>
                  </div>
                </AccordionContent>
              </AccordionItem>

              <AccordionItem value="rl">
                <AccordionTrigger className="flex items-center gap-2">
                  <Target className="w-4 h-4" />
                  RL Learning
                </AccordionTrigger>
                <AccordionContent>
                  <div className="grid grid-cols-2 gap-4 text-sm">
                    <div>
                      <span className="font-medium">RL Learning Rate:</span>
                      <Badge variant="secondary" className="ml-2">{parameters.rl_learning_rate}</Badge>
                    </div>
                    <div>
                      <span className="font-medium">Discount Factor:</span>
                      <Badge variant="secondary" className="ml-2">{parameters.rl_discount_factor}</Badge>
                    </div>
                    <div>
                      <span className="font-medium">Exploration Rate:</span>
                      <Badge variant="secondary" className="ml-2">{parameters.rl_exploration_rate}</Badge>
                    </div>
                  </div>
                </AccordionContent>
              </AccordionItem>
            </Accordion>
          </CardContent>
        </Card>
      )}
    </div>
  )
}

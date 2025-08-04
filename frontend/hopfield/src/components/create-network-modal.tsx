"use client"

import { useState } from "react"
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Textarea } from "@/components/ui/textarea"
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "@/components/ui/accordion"
import { toast } from "@/hooks/use-toast"

interface CreateNetworkModalProps {
  isOpen: boolean
  onClose: () => void
  onNetworkCreated: () => void
}

interface NetworkFormData {
  name: string
  description: string
  // STAG Parameters
  sdr_dimensionality: number
  sdr_sparsity: number
  // GNG Parameters
  max_nodes: number
  n_iter_before_neuron_added: number
  max_edge_age: number
  winner_learning_rate: number
  neighbor_learning_rate: number
  error_decay_rate: number
  // HTM Parameters
  cells_per_column: number
  activation_threshold: number
  initial_permanence: number
  connected_permanence: number
  permanence_increment: number
  permanence_decrement: number
  // RL Parameters
  rl_learning_rate: number
  rl_discount_factor: number
  rl_exploration_rate: number
}

const defaultFormData: NetworkFormData = {
  name: "",
  description: "",
  // STAG Parameters
  sdr_dimensionality: 2048,
  sdr_sparsity: 40,
  // GNG Parameters
  max_nodes: 5000,
  n_iter_before_neuron_added: 100,
  max_edge_age: 50,
  winner_learning_rate: 0.1,
  neighbor_learning_rate: 0.01,
  error_decay_rate: 0.999,
  // HTM Parameters
  cells_per_column: 16,
  activation_threshold: 10,
  initial_permanence: 0.21,
  connected_permanence: 0.5,
  permanence_increment: 0.1,
  permanence_decrement: 0.05,
  // RL Parameters
  rl_learning_rate: 0.1,
  rl_discount_factor: 0.9,
  rl_exploration_rate: 0.9,
}

export function CreateNetworkModal({ isOpen, onClose, onNetworkCreated }: CreateNetworkModalProps) {
  const [formData, setFormData] = useState<NetworkFormData>(defaultFormData)
  const [isCreating, setIsCreating] = useState(false)

  const handleInputChange = (field: keyof NetworkFormData, value: string | number) => {
    setFormData(prev => ({
      ...prev,
      [field]: value
    }))
  }

  const validateForm = (): string | null => {
    if (!formData.name.trim()) {
      return "Please enter a network name."
    }

    if (formData.sdr_sparsity >= formData.sdr_dimensionality) {
      return "SDR sparsity must be less than dimensionality."
    }

    if (formData.sdr_sparsity <= 0 || formData.sdr_dimensionality <= 0) {
      return "SDR parameters must be positive integers."
    }

    if (formData.winner_learning_rate <= 0 || formData.winner_learning_rate > 1) {
      return "Winner learning rate must be between 0 and 1."
    }

    if (formData.neighbor_learning_rate <= 0 || formData.neighbor_learning_rate > 1) {
      return "Neighbor learning rate must be between 0 and 1."
    }

    if (formData.error_decay_rate <= 0 || formData.error_decay_rate > 1) {
      return "Error decay rate must be between 0 and 1."
    }

    if (formData.initial_permanence >= formData.connected_permanence) {
      return "Initial permanence must be less than connected permanence."
    }

    if (formData.rl_learning_rate <= 0 || formData.rl_learning_rate > 1) {
      return "RL learning rate must be between 0 and 1."
    }

    if (formData.rl_discount_factor <= 0 || formData.rl_discount_factor > 1) {
      return "RL discount factor must be between 0 and 1."
    }

    if (formData.rl_exploration_rate < 0 || formData.rl_exploration_rate > 1) {
      return "RL exploration rate must be between 0 and 1."
    }

    return null
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()

    const validationError = validateForm()
    if (validationError) {
      toast({
        title: "Validation Error",
        description: validationError,
        variant: "destructive",
      })
      return
    }

    setIsCreating(true)
    try {
      const response = await fetch("http://localhost:8000/api/brain/networks/", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(formData),
      })

      if (response.ok) {
        toast({
          title: "Network Created",
          description: `Successfully created network "${formData.name}".`,
        })
        setFormData(defaultFormData)
        onNetworkCreated()
        onClose()
      } else {
        const errorData = await response.json()
        throw new Error(errorData.error || "Failed to create network")
      }
    } catch (error) {
      console.error("Error creating network:", error)
      toast({
        title: "Creation Failed",
        description: error instanceof Error ? error.message : "Failed to create network.",
        variant: "destructive",
      })
    } finally {
      setIsCreating(false)
    }
  }

  const resetToDefaults = () => {
    setFormData(prev => ({
      ...defaultFormData,
      name: prev.name,
      description: prev.description
    }))
    toast({
      title: "Reset Complete",
      description: "All parameters have been reset to default values.",
    })
  }

  return (
    <Dialog open={isOpen} onOpenChange={onClose}>
      <DialogContent className="sm:max-w-[600px] max-h-[80vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle>Create New Brain Network</DialogTitle>
        </DialogHeader>
        <form onSubmit={handleSubmit} className="space-y-4">
          {/* Basic Information */}
          <div className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="name">Network Name *</Label>
              <Input
                id="name"
                value={formData.name}
                onChange={(e) => handleInputChange('name', e.target.value)}
                placeholder="Enter network name..."
                required
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="description">Description</Label>
              <Textarea
                id="description"
                value={formData.description}
                onChange={(e) => handleInputChange('description', e.target.value)}
                placeholder="Enter network description..."
                rows={3}
              />
            </div>
          </div>

          {/* Hyperparameters */}
          <Accordion type="multiple" className="w-full">
            {/* STAG Parameters */}
            <AccordionItem value="stag">
              <AccordionTrigger>STAG Parameters</AccordionTrigger>
              <AccordionContent className="space-y-4">
                <div className="grid grid-cols-2 gap-4">
                  <div className="space-y-2">
                    <Label htmlFor="sdr_dimensionality">SDR Dimensionality</Label>
                    <Input
                      id="sdr_dimensionality"
                      type="number"
                      value={formData.sdr_dimensionality}
                      onChange={(e) => handleInputChange('sdr_dimensionality', parseInt(e.target.value))}
                      min="1"
                    />
                    <p className="text-xs text-muted-foreground">Total number of bits in an SDR</p>
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="sdr_sparsity">SDR Sparsity</Label>
                    <Input
                      id="sdr_sparsity"
                      type="number"
                      value={formData.sdr_sparsity}
                      onChange={(e) => handleInputChange('sdr_sparsity', parseInt(e.target.value))}
                      min="1"
                    />
                    <p className="text-xs text-muted-foreground">Number of active bits in an SDR</p>
                  </div>
                </div>
              </AccordionContent>
            </AccordionItem>

            {/* GNG Parameters */}
            <AccordionItem value="gng">
              <AccordionTrigger>Growing Neural Gas Parameters</AccordionTrigger>
              <AccordionContent className="space-y-4">
                <div className="grid grid-cols-2 gap-4">
                  <div className="space-y-2">
                    <Label htmlFor="max_nodes">Max Nodes</Label>
                    <Input
                      id="max_nodes"
                      type="number"
                      value={formData.max_nodes}
                      onChange={(e) => handleInputChange('max_nodes', parseInt(e.target.value))}
                      min="1"
                    />
                    <p className="text-xs text-muted-foreground">Maximum number of nodes</p>
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="n_iter_before_neuron_added">Iterations Before New Neuron</Label>
                    <Input
                      id="n_iter_before_neuron_added"
                      type="number"
                      value={formData.n_iter_before_neuron_added}
                      onChange={(e) => handleInputChange('n_iter_before_neuron_added', parseInt(e.target.value))}
                      min="1"
                    />
                    <p className="text-xs text-muted-foreground">Iterations before adding new node</p>
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="max_edge_age">Max Edge Age</Label>
                    <Input
                      id="max_edge_age"
                      type="number"
                      value={formData.max_edge_age}
                      onChange={(e) => handleInputChange('max_edge_age', parseInt(e.target.value))}
                      min="1"
                    />
                    <p className="text-xs text-muted-foreground">Maximum age before edge pruning</p>
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="winner_learning_rate">Winner Learning Rate</Label>
                    <Input
                      id="winner_learning_rate"
                      type="number"
                      step="0.01"
                      value={formData.winner_learning_rate}
                      onChange={(e) => handleInputChange('winner_learning_rate', parseFloat(e.target.value))}
                      min="0"
                      max="1"
                    />
                    <p className="text-xs text-muted-foreground">Learning rate for winning node</p>
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="neighbor_learning_rate">Neighbor Learning Rate</Label>
                    <Input
                      id="neighbor_learning_rate"
                      type="number"
                      step="0.01"
                      value={formData.neighbor_learning_rate}
                      onChange={(e) => handleInputChange('neighbor_learning_rate', parseFloat(e.target.value))}
                      min="0"
                      max="1"
                    />
                    <p className="text-xs text-muted-foreground">Learning rate for neighbors</p>
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="error_decay_rate">Error Decay Rate</Label>
                    <Input
                      id="error_decay_rate"
                      type="number"
                      step="0.001"
                      value={formData.error_decay_rate}
                      onChange={(e) => handleInputChange('error_decay_rate', parseFloat(e.target.value))}
                      min="0"
                      max="1"
                    />
                    <p className="text-xs text-muted-foreground">Decay factor for node errors</p>
                  </div>
                </div>
              </AccordionContent>
            </AccordionItem>

            {/* HTM Parameters */}
            <AccordionItem value="htm">
              <AccordionTrigger>Hierarchical Temporal Memory Parameters</AccordionTrigger>
              <AccordionContent className="space-y-4">
                <div className="grid grid-cols-2 gap-4">
                  <div className="space-y-2">
                    <Label htmlFor="cells_per_column">Cells Per Column</Label>
                    <Input
                      id="cells_per_column"
                      type="number"
                      value={formData.cells_per_column}
                      onChange={(e) => handleInputChange('cells_per_column', parseInt(e.target.value))}
                      min="1"
                    />
                    <p className="text-xs text-muted-foreground">Number of cells within each column</p>
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="activation_threshold">Activation Threshold</Label>
                    <Input
                      id="activation_threshold"
                      type="number"
                      value={formData.activation_threshold}
                      onChange={(e) => handleInputChange('activation_threshold', parseInt(e.target.value))}
                      min="1"
                    />
                    <p className="text-xs text-muted-foreground">Connected synapses needed for activation</p>
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="initial_permanence">Initial Permanence</Label>
                    <Input
                      id="initial_permanence"
                      type="number"
                      step="0.01"
                      value={formData.initial_permanence}
                      onChange={(e) => handleInputChange('initial_permanence', parseFloat(e.target.value))}
                      min="0"
                      max="1"
                    />
                    <p className="text-xs text-muted-foreground">Initial permanence for new synapses</p>
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="connected_permanence">Connected Permanence</Label>
                    <Input
                      id="connected_permanence"
                      type="number"
                      step="0.01"
                      value={formData.connected_permanence}
                      onChange={(e) => handleInputChange('connected_permanence', parseFloat(e.target.value))}
                      min="0"
                      max="1"
                    />
                    <p className="text-xs text-muted-foreground">Permanence threshold for connection</p>
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="permanence_increment">Permanence Increment</Label>
                    <Input
                      id="permanence_increment"
                      type="number"
                      step="0.01"
                      value={formData.permanence_increment}
                      onChange={(e) => handleInputChange('permanence_increment', parseFloat(e.target.value))}
                      min="0"
                      max="1"
                    />
                    <p className="text-xs text-muted-foreground">Increase for active synapses</p>
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="permanence_decrement">Permanence Decrement</Label>
                    <Input
                      id="permanence_decrement"
                      type="number"
                      step="0.01"
                      value={formData.permanence_decrement}
                      onChange={(e) => handleInputChange('permanence_decrement', parseFloat(e.target.value))}
                      min="0"
                      max="1"
                    />
                    <p className="text-xs text-muted-foreground">Decrease for inactive synapses</p>
                  </div>
                </div>
              </AccordionContent>
            </AccordionItem>

            {/* RL Parameters */}
            <AccordionItem value="rl">
              <AccordionTrigger>Reinforcement Learning Parameters</AccordionTrigger>
              <AccordionContent className="space-y-4">
                <div className="grid grid-cols-2 gap-4">
                  <div className="space-y-2">
                    <Label htmlFor="rl_learning_rate">RL Learning Rate</Label>
                    <Input
                      id="rl_learning_rate"
                      type="number"
                      step="0.01"
                      value={formData.rl_learning_rate}
                      onChange={(e) => handleInputChange('rl_learning_rate', parseFloat(e.target.value))}
                      min="0"
                      max="1"
                    />
                    <p className="text-xs text-muted-foreground">Q-learning learning rate (alpha)</p>
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="rl_discount_factor">RL Discount Factor</Label>
                    <Input
                      id="rl_discount_factor"
                      type="number"
                      step="0.01"
                      value={formData.rl_discount_factor}
                      onChange={(e) => handleInputChange('rl_discount_factor', parseFloat(e.target.value))}
                      min="0"
                      max="1"
                    />
                    <p className="text-xs text-muted-foreground">Discount factor for future rewards (gamma)</p>
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="rl_exploration_rate">RL Exploration Rate</Label>
                    <Input
                      id="rl_exploration_rate"
                      type="number"
                      step="0.01"
                      value={formData.rl_exploration_rate}
                      onChange={(e) => handleInputChange('rl_exploration_rate', parseFloat(e.target.value))}
                      min="0"
                      max="1"
                    />
                    <p className="text-xs text-muted-foreground">Exploration rate for random actions (epsilon)</p>
                  </div>
                </div>
              </AccordionContent>
            </AccordionItem>
          </Accordion>

          <div className="flex justify-between">
            <Button type="button" variant="outline" onClick={resetToDefaults}>
              Reset to Defaults
            </Button>
            <div className="flex space-x-2">
              <Button type="button" variant="outline" onClick={onClose}>
                Cancel
              </Button>
              <Button type="submit" disabled={isCreating}>
                {isCreating ? "Creating..." : "Create Network"}
              </Button>
            </div>
          </div>
        </form>
      </DialogContent>
    </Dialog>
  )
}

"use client"

import { useState } from "react"
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from "@/components/ui/dialog"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Slider } from "@/components/ui/slider"
import { Loader2 } from "lucide-react"
import { toast } from "@/hooks/use-toast"

interface CreateNetworkModalProps {
  isOpen: boolean
  onClose: () => void
  onNetworkCreated: (network: any) => void
}

export function CreateNetworkModal({ isOpen, onClose, onNetworkCreated }: CreateNetworkModalProps) {
  const [name, setName] = useState("")
  const [embeddingDim, setEmbeddingDim] = useState([64])
  const [beta, setBeta] = useState([20.0])
  const [learningRate, setLearningRate] = useState([0.1])
  const [isCreating, setIsCreating] = useState(false)

  const handleCreate = async () => {
    if (!name.trim()) {
      toast({
        title: "Error",
        description: "Please enter a network name",
        variant: "destructive",
      })
      return
    }

    setIsCreating(true)
    try {
      const response = await fetch("http://localhost:8000/api/networks/", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          name: name.trim(),
          embedding_dim: embeddingDim[0],
          beta: beta[0],
          learning_rate: learningRate[0],
        }),
      })

      if (response.ok) {
        const network = await response.json()
        onNetworkCreated(network)
        toast({
          title: "Success",
          description: `Network "${network.name}" created successfully`,
        })
        // Reset form
        setName("")
        setEmbeddingDim([64])
        setBeta([20.0])
        setLearningRate([0.1])
      } else {
        throw new Error("Failed to create network")
      }
    } catch (error) {
      toast({
        title: "Error",
        description: "Failed to create network",
        variant: "destructive",
      })
    } finally {
      setIsCreating(false)
    }
  }

  return (
    <Dialog open={isOpen} onOpenChange={onClose}>
      <DialogContent className="sm:max-w-[425px]">
        <DialogHeader>
          <DialogTitle>Create New Neural Network</DialogTitle>
          <DialogDescription>Configure the parameters for your new Hopfield network</DialogDescription>
        </DialogHeader>

        <div className="space-y-6 py-4">
          <div className="space-y-2">
            <Label htmlFor="name">Network Name</Label>
            <Input
              id="name"
              placeholder="Enter network name..."
              value={name}
              onChange={(e) => setName(e.target.value)}
            />
          </div>

          <div className="space-y-2">
            <Label>Embedding Dimension: {embeddingDim[0]}</Label>
            <Slider
              value={embeddingDim}
              onValueChange={setEmbeddingDim}
              max={512}
              min={32}
              step={32}
              className="w-full"
            />
            <div className="flex justify-between text-xs text-gray-500">
              <span>32</span>
              <span>512</span>
            </div>
          </div>

          <div className="space-y-2">
            <Label>Beta (Temperature): {beta[0]}</Label>
            <Slider value={beta} onValueChange={setBeta} max={50} min={1} step={1} className="w-full" />
            <div className="flex justify-between text-xs text-gray-500">
              <span>1</span>
              <span>50</span>
            </div>
          </div>

          <div className="space-y-2">
            <Label>Learning Rate: {learningRate[0]}</Label>
            <Slider
              value={learningRate}
              onValueChange={setLearningRate}
              max={1}
              min={0.01}
              step={0.01}
              className="w-full"
            />
            <div className="flex justify-between text-xs text-gray-500">
              <span>0.01</span>
              <span>1.0</span>
            </div>
          </div>
        </div>

        <div className="flex justify-end space-x-2">
          <Button variant="outline" onClick={onClose} disabled={isCreating}>
            Cancel
          </Button>
          <Button onClick={handleCreate} disabled={isCreating}>
            {isCreating ? (
              <>
                <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                Creating...
              </>
            ) : (
              "Create Network"
            )}
          </Button>
        </div>
      </DialogContent>
    </Dialog>
  )
}

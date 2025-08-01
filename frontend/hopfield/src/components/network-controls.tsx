"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Label } from "@/components/ui/label"
import { Textarea } from "@/components/ui/textarea"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Save, Search, Loader2 } from "lucide-react"
import { toast } from "@/hooks/use-toast"

interface NetworkControlsProps {
  networkId: number
}

export function NetworkControls({ networkId }: NetworkControlsProps) {
  const [storeText, setStoreText] = useState("")
  const [retrieveText, setRetrieveText] = useState("")
  const [isStoring, setIsStoring] = useState(false)
  const [isRetrieving, setIsRetrieving] = useState(false)
  const [retrievalResult, setRetrievalResult] = useState<any>(null)

  const handleStore = async () => {
    if (!storeText.trim()) {
      toast({
        title: "Error",
        description: "Please enter text to store",
        variant: "destructive",
      })
      return
    }

    setIsStoring(true)
    try {
      const response = await fetch(`http://localhost:8000/api/networks/${networkId}/store_pattern/`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ text: storeText }),
      })

      const result = await response.json()

      if (result.success) {
        toast({
          title: "Pattern Stored",
          description: `Successfully stored pattern. Total patterns: ${result.total_patterns}`,
        })
        setStoreText("")
      } else {
        toast({
          title: "Storage Failed",
          description: result.message,
          variant: "destructive",
        })
      }
    } catch (error) {
      toast({
        title: "Error",
        description: "Failed to store pattern",
        variant: "destructive",
      })
    } finally {
      setIsStoring(false)
    }
  }

  const handleRetrieve = async () => {
    if (!retrieveText.trim()) {
      toast({
        title: "Error",
        description: "Please enter text to retrieve",
        variant: "destructive",
      })
      return
    }

    setIsRetrieving(true)
    try {
      const response = await fetch(`http://localhost:8000/api/networks/${networkId}/retrieve_pattern/`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          query_text: retrieveText,
          max_iter: 10,
        }),
      })

      const result = await response.json()

      if (result.success) {
        setRetrievalResult(result)
        toast({
          title: "Pattern Retrieved",
          description: `Found pattern with ${(result.confidence_score * 100).toFixed(1)}% confidence`,
        })
      } else {
        toast({
          title: "Retrieval Failed",
          description: result.message,
          variant: "destructive",
        })
      }
    } catch (error) {
      toast({
        title: "Error",
        description: "Failed to retrieve pattern",
        variant: "destructive",
      })
    } finally {
      setIsRetrieving(false)
    }
  }

  return (
    <Tabs defaultValue="store" className="w-full">
      <TabsList className="grid w-full grid-cols-2">
        <TabsTrigger value="store">Store Pattern</TabsTrigger>
        <TabsTrigger value="retrieve">Retrieve Pattern</TabsTrigger>
      </TabsList>

      <TabsContent value="store" className="space-y-4">
        <div className="space-y-2">
          <Label htmlFor="store-text">Text to Store</Label>
          <Textarea
            id="store-text"
            placeholder="Enter text to store in the neural network..."
            value={storeText}
            onChange={(e) => setStoreText(e.target.value)}
            rows={4}
            className="resize-none"
          />
        </div>

        <Button
          onClick={handleStore}
          disabled={isStoring || !storeText.trim()}
          className="w-full bg-blue-600 hover:bg-blue-700"
        >
          {isStoring ? (
            <>
              <Loader2 className="w-4 h-4 mr-2 animate-spin" />
              Storing...
            </>
          ) : (
            <>
              <Save className="w-4 h-4 mr-2" />
              Store Pattern
            </>
          )}
        </Button>
      </TabsContent>

      <TabsContent value="retrieve" className="space-y-4">
        <div className="space-y-2">
          <Label htmlFor="retrieve-text">Query Text</Label>
          <Textarea
            id="retrieve-text"
            placeholder="Enter text to retrieve similar patterns..."
            value={retrieveText}
            onChange={(e) => setRetrieveText(e.target.value)}
            rows={3}
            className="resize-none"
          />
        </div>

        <Button
          onClick={handleRetrieve}
          disabled={isRetrieving || !retrieveText.trim()}
          className="w-full bg-purple-600 hover:bg-purple-700"
        >
          {isRetrieving ? (
            <>
              <Loader2 className="w-4 h-4 mr-2 animate-spin" />
              Retrieving...
            </>
          ) : (
            <>
              <Search className="w-4 h-4 mr-2" />
              Retrieve Pattern
            </>
          )}
        </Button>
      </TabsContent>
    </Tabs>
  )
}

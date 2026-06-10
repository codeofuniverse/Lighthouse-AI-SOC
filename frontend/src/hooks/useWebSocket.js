import { useEffect, useRef, useCallback, useState } from 'react'
import { isBackendOffline } from '../lib/api'

const WS_URL = `${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws/alerts`
const MAX_BACKOFF = 120_000

export function useWebSocket(onAlert) {
  const [connected, setConnected] = useState(false)
  const wsRef = useRef(null)
  const backoffRef = useRef(1000)
  const timerRef = useRef(null)
  const mountedRef = useRef(true)
  const onAlertRef = useRef(onAlert)
  const failedAttemptsRef = useRef(0)

  useEffect(() => { onAlertRef.current = onAlert }, [onAlert])

  const connect = useCallback(() => {
    if (!mountedRef.current) return
    if (isBackendOffline() && failedAttemptsRef.current >= 1) return

    try {
      const ws = new WebSocket(WS_URL)
      wsRef.current = ws

      ws.onopen = () => {
        if (!mountedRef.current) return
        setConnected(true)
        backoffRef.current = 1000
        failedAttemptsRef.current = 0
      }

      ws.onmessage = (evt) => {
        if (!mountedRef.current) return
        try {
          const alert = JSON.parse(evt.data)
          onAlertRef.current?.(alert)
        } catch {
          // malformed message — ignore
        }
      }

      ws.onclose = () => {
        if (!mountedRef.current) return
        setConnected(false)
        failedAttemptsRef.current += 1
        timerRef.current = setTimeout(() => {
          backoffRef.current = Math.min(backoffRef.current * 2, MAX_BACKOFF)
          connect()
        }, backoffRef.current)
      }

      ws.onerror = () => {
        ws.close()
      }
    } catch {
      // WebSocket constructor failed (e.g. no backend)
      setConnected(false)
      failedAttemptsRef.current += 1
      timerRef.current = setTimeout(() => {
        backoffRef.current = Math.min(backoffRef.current * 2, MAX_BACKOFF)
        connect()
      }, backoffRef.current)
    }
  }, [])

  useEffect(() => {
    mountedRef.current = true
    connect()
    return () => {
      mountedRef.current = false
      clearTimeout(timerRef.current)
      wsRef.current?.close()
    }
  }, [connect])

  return { connected }
}

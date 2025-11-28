import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { cn } from '@/lib/utils';

interface CoolTooltipProps {
  content: React.ReactNode;
  children: React.ReactNode;
  className?: string;
  side?: 'top' | 'bottom' | 'left' | 'right';
}

export const CoolTooltip = ({ 
  content, 
  children, 
  className,
  side = 'top'
}: CoolTooltipProps) => {
  const [isVisible, setIsVisible] = useState(false);

  const variants = {
    hidden: { opacity: 0, scale: 0.9, y: 5 },
    visible: { 
      opacity: 1, 
      scale: 1, 
      y: 0,
      transition: {
        type: 'spring',
        stiffness: 400,
        damping: 20
      }
    },
    exit: { 
      opacity: 0, 
      scale: 0.9, 
      y: 5,
      transition: {
        duration: 0.15
      }
    }
  };

  const getPosition = () => {
    switch (side) {
      case 'top': return '-top-3 left-1/2 -translate-x-1/2 -translate-y-full';
      case 'bottom': return '-bottom-3 left-1/2 -translate-x-1/2 translate-y-full';
      case 'left': return '-left-3 top-1/2 -translate-y-1/2 -translate-x-full';
      case 'right': return '-right-3 top-1/2 -translate-y-1/2 translate-x-full';
      default: return '-top-3 left-1/2 -translate-x-1/2 -translate-y-full';
    }
  };

  return (
    <div 
      className="relative inline-block"
      onMouseEnter={() => setIsVisible(true)}
      onMouseLeave={() => setIsVisible(false)}
    >
      {children}
      <AnimatePresence>
        {isVisible && (
          <motion.div
            initial="hidden"
            animate="visible"
            exit="exit"
            variants={variants}
            className={cn(
              "absolute z-[100] px-4 py-2 text-sm font-medium text-gray-800 bg-white rounded-2xl shadow-[0_4px_20px_-2px_rgba(0,0,0,0.15)] whitespace-nowrap pointer-events-none",
              "border border-gray-100/50",
              getPosition(),
              className
            )}
          >
            <span className="relative z-10 flex items-center gap-2">{content}</span>
            
            {/* Chat Bubble Arrow */}
            {side === 'top' && (
              <div className="absolute -bottom-1.5 left-1/2 -translate-x-1/2 w-3 h-3 bg-white rotate-45 border-r border-b border-gray-100/50 rounded-[1px]" />
            )}
            {side === 'bottom' && (
              <div className="absolute -top-1.5 left-1/2 -translate-x-1/2 w-3 h-3 bg-white rotate-45 border-l border-t border-gray-100/50 rounded-[1px]" />
            )}
            {side === 'left' && (
              <div className="absolute -right-1.5 top-1/2 -translate-y-1/2 w-3 h-3 bg-white rotate-45 border-r border-t border-gray-100/50 rounded-[1px]" />
            )}
             {side === 'right' && (
              <div className="absolute -left-1.5 top-1/2 -translate-y-1/2 w-3 h-3 bg-white rotate-45 border-l border-b border-gray-100/50 rounded-[1px]" />
            )}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};
